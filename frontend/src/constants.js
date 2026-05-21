export const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'
export const WS_BASE  = import.meta.env.VITE_WS_URL  || 'ws://localhost:8000'

export const PATHOLOGIES = [
  'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
  'Effusion', 'Emphysema', 'Fibrosis', 'Hernia',
  'Infiltration', 'Mass', 'Nodule', 'Pleural_Thickening',
  'Pneumonia', 'Pneumothorax',
]
