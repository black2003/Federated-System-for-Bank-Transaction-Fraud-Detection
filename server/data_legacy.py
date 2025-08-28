# server/data.py (renamed to avoid package shadowing)
import random
import string
import time

# Parameters: step, type, amount, nameOrig, oldbalanceOrg, newbalanceOrig,
#             nameDest, oldbalanceDest, newbalanceDest, isFraud, isFlaggedFraud

TX_TYPES = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT"]

def _random_name(prefix="C"):
    return prefix + ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))

def generate_transaction(step:int):
    amount = round(random.uniform(10, 5000), 2)
    oldbalanceOrg = round(random.uniform(0, 10000), 2)
    newbalanceOrg = max(0, oldbalanceOrg - amount)
    oldbalanceDest = round(random.uniform(0, 10000), 2)
    newbalanceDest = oldbalanceDest + amount

    # Fraud logic (very naive example)
    isFraud = int(amount > 4000 and random.random() < 0.3)
    isFlaggedFraud = int(isFraud and random.random() < 0.5)

    return {
        "step": step,
        "type": random.choice(TX_TYPES),
        "amount": amount,
        "nameOrig": _random_name("C"),
        "oldbalanceOrg": oldbalanceOrg,
        "newbalanceOrig": newbalanceOrg,
        "nameDest": _random_name("M"),
        "oldbalanceDest": oldbalanceDest,
        "newbalanceDest": newbalanceDest,
        "isFraud": isFraud,
        "isFlaggedFraud": isFlaggedFraud,
    }

def stream():
    """Infinite generator for synthetic transactions"""
    step = 0
    while True:
        yield generate_transaction(step)
        step += 1
        time.sleep(1)  # 1 tx/sec


