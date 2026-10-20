
from src.modeling import train_and_evaluate
from src.validation import load_and_clean

df, report = load_and_clean("data/AI_A2_8149.csv")

print("Dataset validation report:", report)

summary = train_and_evaluate(df, seed=8149)

print("\nMODEL TRAINING COMPLETED")
print(summary)
