import sys
from crew import ChestXRAICrew


def run():
    image_path = sys.argv[1] if len(sys.argv) > 1 else "sample.jpg"
    inputs = {"image_path": image_path}
    result = ChestXRAICrew().crew().kickoff(inputs=inputs)

    print("\n" + "=" * 60)
    print("TRIAGE REPORT")
    print("=" * 60)
    print(result.raw)

    print("\n" + "=" * 60)
    decision = input("Physician Review — Approve report? (yes/no/revise): ").strip().lower()

    if decision == "yes":
        print("Report approved.")
    elif decision == "revise":
        feedback = input("Enter revision notes: ")
        print(f"Revision requested: {feedback}")
    else:
        print("Report rejected.")


if __name__ == "__main__":
    run()