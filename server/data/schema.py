from dataclasses import dataclass

@dataclass
class Transaction:
    step: int
    type: str
    amount: float
    nameOrig: str
    oldbalanceOrg: float
    newbalanceOrig: float
    nameDest: str
    oldbalanceDest: float
    newbalanceDest: float
    isFraud: int
    isFlaggedFraud: int

    @staticmethod
    def fields():
        return [
            "step","type","amount","nameOrig","oldbalanceOrg","newbalanceOrig",
            "nameDest","oldbalanceDest","newbalanceDest","isFraud","isFlaggedFraud"
        ]
