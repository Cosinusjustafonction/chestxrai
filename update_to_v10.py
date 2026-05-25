import json
import re

notebook_path = '/Users/elmesbahiilyass/Desktop/Projects/4thyear/DLproject/chestxrai/herdimagechestray.ipynb'
out_path = '/Users/elmesbahiilyass/Desktop/Projects/4thyear/DLproject/chestxrai/herdimagechestray_v10.ipynb'

with open(notebook_path, 'r') as f:
    nb = json.load(f)

NEW_ESCAPE_CODE = """def get_param_vector(model):
    return torch.cat([p.data.view(-1) for p in model.parameters() if p.requires_grad])

def compute_train_loss_batches(model, loss_fn, loader, device, num_batches=5):
    \"\"\"Deterministic loss on a few batches (eval mode).\"\"\"
    model.eval()
    total_loss = 0.0
    count = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            total_loss += loss_fn(model(x), y).item()
            count += 1
            if count >= num_batches:
                break
    return total_loss / count

def probe_basin(model, original_state, direction, distance, loss_fn, loader, device, probe_steps=10, probe_lr=1e-3, label="Probe"):
    \"\"\"
    Jump to a new point and do a fast local descent to estimate the basin floor.
    \"\"\"
    import copy
    # 1. Jump to the new point
    model.load_state_dict(copy.deepcopy(original_state))
    params = [p for p in model.parameters() if p.requires_grad]
    
    with torch.no_grad():
        for p, vi in zip(params, direction):
            p.add_(vi, alpha=distance)
            
    jump_loss = compute_train_loss_batches(model, loss_fn, loader, device, num_batches=8)
    pre_probe_params = get_param_vector(model)
    
    # 2. Fast descent (Probe)
    probe_optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=probe_lr)
    
    model.train()
    step_count = 0
    # Use a small number of batches for the probe descent
    for x, y in loader:
        if step_count >= probe_steps:
            break
        x, y = x.to(device), y.to(device)
        probe_optimizer.zero_grad()
        with autocast("cuda"):
            logits = model(x)
            loss = loss_fn(logits, y)
        
        # We don't strictly need a scaler here if it's just a rough probe
        loss.backward()
        probe_optimizer.step()
        step_count += 1

    # 3. Evaluate the basin floor
    floor_loss = compute_train_loss_batches(model, loss_fn, loader, device, num_batches=8)
    floor_state = copy.deepcopy(model.state_dict())
    
    # Distance from original state to the floor of the new basin
    # Avoid gradient tracking issues by using tensors directly
    with torch.no_grad():
        original_vector = torch.cat([p.view(-1) for p in original_state.values() if p.is_floating_point()])
        floor_vector = get_param_vector(model)
        dist_from_origin = torch.norm(floor_vector - original_vector).item()
    
    print(f"          [{label}] Jump Dist: {distance:.4f} | Jump Loss: {jump_loss:.4f} | Floor Loss: {floor_loss:.4f} | Dist from Origin: {dist_from_origin:.4f}")
    
    return {
        'jump_loss': jump_loss,
        'floor_loss': floor_loss,
        'dist_from_origin': dist_from_origin,
        'floor_state': floor_state
    }

def herd_v10_probe_escape(model, loss_fn, loader, device, lanczos_k=15,
                          max_directions=3, jump_scales=[0.5, 1.0, 2.0], probe_steps=10):
    \"\"\"
    HERD v10: Look-Ahead Basin Probing.
    Instead of walking, it jumps and probes to find strictly better basins.
    \"\"\"
    import copy
    original_state = copy.deepcopy(model.state_dict())

    log = {
        'has_negative_curvature': False,
        'n_negative_directions': 0,
        'probes': [],
        'escaped': False,
    }

    print(f"\\n    {'='*60}")
    print(f"    HERD v10 PROBE — Lanczos Decomposition")
    print(f"    {'='*60}")
    t_lanczos = time.time()

    eigenvalues, eigvec_coeffs, Q_list, spectrum_info = lanczos_decomposition(
        model, loss_fn, loader, device, k=lanczos_k, num_batches=3
    )

    log['spectrum'] = spectrum_info
    print(f"    Lanczos completed in {time.time() - t_lanczos:.1f}s")
    
    neg_eigvecs = extract_negative_eigenvectors(eigenvalues, eigvec_coeffs, Q_list)
    log['n_negative_directions'] = len(neg_eigvecs)

    del Q_list, eigvec_coeffs, eigenvalues
    torch.cuda.empty_cache() if device.type == 'cuda' else None

    if len(neg_eigvecs) == 0:
        log['has_negative_curvature'] = False
        log['reason'] = 'no_negative_curvature'
        print(f"\\n    No negative curvature found — landscape is a bowl. Continuing standard training.")
        return log

    log['has_negative_curvature'] = True
    start_loss = compute_train_loss_batches(model, loss_fn, loader, device, num_batches=8)
    print(f"\\n    Original Basin Loss: {start_loss:.5f}")
    
    print(f"\\n    {'='*60}")
    print(f"    PROBING ATTEMPTS")
    print(f"    {'='*60}")

    n_to_try = min(len(neg_eigvecs), max_directions)
    
    best_probe_loss = start_loss - 0.005 # We want a STRICTLY better basin to commit
    best_probe_state = None
    best_probe_info = None

    for dir_idx in range(n_to_try):
        direction, eig_val = neg_eigvecs[dir_idx]
        base_dist = 1.0 / np.sqrt(abs(eig_val) + 1e-10)
        base_dist = np.clip(base_dist, 0.01, 1.0) # limit base jump
        
        print(f"\\n      {'─'*50}")
        print(f"      Direction {dir_idx+1}/{n_to_try}: lambda = {eig_val:.4f} (Base dist: {base_dist:.4f})")
        print(f"      {'─'*50}")

        for scale in jump_scales:
            dist = base_dist * scale
            
            # Probe Positive
            pos_info = probe_basin(model, original_state, direction, dist, loss_fn, loader, device, 
                                   probe_steps=probe_steps, label=f"D{dir_idx+1} +{scale}x")
            log['probes'].append(pos_info)
            if pos_info['floor_loss'] < best_probe_loss:
                best_probe_loss = pos_info['floor_loss']
                best_probe_state = pos_info['floor_state']
                best_probe_info = pos_info
                
            # Probe Negative
            neg_direction = [-d for d in direction]
            neg_info = probe_basin(model, original_state, neg_direction, dist, loss_fn, loader, device, 
                                   probe_steps=probe_steps, label=f"D{dir_idx+1} -{scale}x")
            log['probes'].append(neg_info)
            if neg_info['floor_loss'] < best_probe_loss:
                best_probe_loss = neg_info['floor_loss']
                best_probe_state = neg_info['floor_state']
                best_probe_info = neg_info

    # Final decision
    print(f"\\n    {'='*60}")
    print(f"    PROBE VERDICT")
    print(f"    {'='*60}")

    if best_probe_state is not None:
        model.load_state_dict(best_probe_state)
        log['escaped'] = True
        log['reason'] = 'better_basin_found'
        print(f"    SUCCESS: Teleporting to better basin!")
        print(f"      Original Loss: {start_loss:.5f}")
        print(f"      New Basin Floor: {best_probe_loss:.5f}")
        print(f"      Improvement: {start_loss - best_probe_loss:.5f}")
    else:
        model.load_state_dict(original_state)
        log['escaped'] = False
        log['reason'] = 'no_better_basin'
        print(f"    FAILED: No better basins found nearby.")
        print(f"    Model restored to original basin (Loss: {start_loss:.5f})")

    print(f"    {'='*60}\\n")
    return log
"""

# Modify the cells
for cell in nb['cells']:
    if cell['cell_type'] == 'markdown':
        if 'HERD v9' in ''.join(cell['source']):
            cell['source'] = [s.replace('HERD v9', 'HERD v10').replace('Deep Escape', 'Look-Ahead Probe') for s in cell['source']]
    
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        
        # Replace the escape functions
        if 'def escape_along_direction' in source:
            # We split the source lines and format properly for jupyter
            lines = [line + '\\n' for line in NEW_ESCAPE_CODE.split('\\n')]
            lines[-1] = lines[-1].rstrip('\\n')
            cell['source'] = lines
            
        # Replace the execution call
        if 'herd_v9_escape' in source and 'escape_log =' in source:
            source = source.replace('HERD v9', 'HERD v10')
            source = source.replace('Deep Escape', 'Look-Ahead Probe')
            source = source.replace('herd_v9_escape', 'herd_v10_probe_escape')
            source = source.replace('min_distance=HERD_MIN_DISTANCE,', 'jump_scales=[0.5, 1.0, 2.0], probe_steps=10,')
            source = source.replace('max_steps=HERD_MAX_ESCAPE_STEPS,', '')
            source = source.replace('loss_patience=HERD_LOSS_PATIENCE,', '')
            source = source.replace('herd_v9_nih_results', 'herd_v10_nih_results')
            source = source.replace('best_densenet121_nih_herd_v9.pth', 'best_densenet121_nih_herd_v10.pth')
            
            lines = [line + '\\n' for line in source.split('\\n')]
            lines[-1] = lines[-1].rstrip('\\n')
            cell['source'] = lines

with open(out_path, 'w') as f:
    json.dump(nb, f, indent=1)

print(f"Successfully generated {out_path}")
