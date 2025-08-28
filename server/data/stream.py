import asyncio
import random
import string
from typing import AsyncIterator, Dict, List

TYPES = ["PAYMENT", "TRANSFER", "CASH_OUT", "DEBIT", "CASH_IN"]

def _rand_name(prefix: str) -> str:
    return prefix + "_" + "".join(random.choices(string.ascii_uppercase + string.digits, k=6))

def synth_txn(step: int) -> Dict:
    ttype = random.choice(TYPES)
    amount = max(1.0, random.expovariate(1/5000))
    old_org = max(0.0, random.gauss(20000, 8000))
    new_org = max(0.0, old_org - amount)
    old_dst = max(0.0, random.gauss(25000, 10000))
    new_dst = old_dst + amount

    is_fraud = 1 if (ttype in ["TRANSFER", "CASH_OUT"] and random.random() < 0.02 and amount > 5000) else 0
    is_flagged = 1 if (is_fraud and amount > 200000) else 0

    return {
        "step": step,
        "type": ttype,
        "amount": float(amount),
        "nameOrig": _rand_name("C"),
        "oldbalanceOrg": float(old_org),
        "newbalanceOrig": float(new_org),
        "nameDest": _rand_name("M"),
        "oldbalanceDest": float(old_dst),
        "newbalanceDest": float(new_dst),
        "isFraud": int(is_fraud),
        "isFlaggedFraud": int(is_flagged),
    }

async def stream_transactions(rate_hz: float = 5.0) -> AsyncIterator[Dict]:
    step = 1
    delay = 1.0 / max(0.1, rate_hz)
    while True:
        yield synth_txn(step)
        step += 1
        await asyncio.sleep(delay)

def generate_batch(n: int, start_step: int = 1) -> List[Dict]:
    return [synth_txn(start_step + i) for i in range(n)]
