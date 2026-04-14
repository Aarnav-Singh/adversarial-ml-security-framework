from src.risk_engine.network_classifier import NetworkRiskClassifier

def build_model(input_dim=41):
    """
    Factory function to build the NetworkRiskClassifier model.
    """
    return NetworkRiskClassifier(input_dim=input_dim)
