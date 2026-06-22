
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.predictor import predict, model  # explicit src. prefix

FEATURE_NAMES = [
    "Age", "Experience", "Income",
    "Family", "Education",
    "Mortgage", "CD Account"
]
def loan_agent(data):
    raw = predict(data)
    print("Number of values returned:", len(raw))
    print("Values:", raw)
    prediction, shap_values, data_array = raw  # unpack after checking
    proba = model.predict_proba(data_array)[0]
    confidence = round(float(max(proba)) * 100, 1)

    insights = []
    suggestions = []

    for i, name in enumerate(FEATURE_NAMES):
        if shap_values[i] > 0:
            insights.append(f"{name} increased approval chance")
        else:
            insights.append(f"{name} decreased approval chance")
            if prediction == 0:  # only suggest fixes on rejection
                suggestions.append(f"Consider improving {name}")

    decision = "✅ Loan Approved" if prediction == 1 else "❌ Loan Rejected"

    return {
        "decision": decision,
        "confidence": confidence,
        "insights": insights,
        "suggestions": suggestions,
        "shap_values": shap_values,
        "data_array": data_array
    }
print("success")