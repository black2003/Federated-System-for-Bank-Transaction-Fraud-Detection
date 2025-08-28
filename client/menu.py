import numpy as np
from .client_node import ClientSession

async def client_menu(client_id: str, name: str, server_app):
    print(f"\n== Client '{client_id}' ({name}) ==")
    session = ClientSession(client_id, name, server_app, input_dim=server_app.global_model.input_dim)

    while True:
        print(
            "\nClient Menu:"
            "\n 1) Pull N new transactions into buffer"
            "\n 2) Predict with GLOBAL model on last pulled batch"
            "\n 3) Predict with LOCAL model on last pulled batch"
            "\n 4) Local retrain once (one epoch) on buffered labeled data"
            "\n 5) Push local update to GLOBAL (FedAvg + IF refresh)"
            "\n 6) Access Admin Menu (requires admin password)"
            "\n 7) Logout"
        )
        choice = input("> ").strip()

        if choice == "1":
            try:
                n = int(input("How many transactions to pull? ").strip())
            except ValueError:
                print("Invalid number.")
                continue
            X, y = session.pull_batch(n=n)
            print(f"Pulled {len(X)} transactions into buffer. Positives in batch: {int(y.sum())}")
            # keep last batch for prediction demo
            session._lastX = X
            session._lasty = y

        elif choice == "2":
            if not hasattr(session, "_lastX"):
                print("Pull a batch first (option 1).")
                continue
            out = session.predict_global(session._lastX)
            show_scores(out, session._lasty, label="GLOBAL")

        elif choice == "3":
            if not hasattr(session, "_lastX"):
                print("Pull a batch first (option 1).")
                continue
            out = session.predict_local(session._lastX)
            show_scores(out, session._lasty, label="LOCAL")

        elif choice == "4":
            msg = session.local_retrain_once()
            print(msg)

        elif choice == "5":
            if not session.has_locally_retrained:
                print("You haven't retrained locally this round (option 4). You can still push, but update is same as global.")
            pkg = session.package_update()
            # FedAvg on LSTM
            server_app.global_model.aggregate_lstm([pkg["lstm_state"]])
            # Update IF using client's latest buffer as background
            server_app.global_model.update_iso(pkg["if_snapshot"])
            # Save global LSTM
            server_app.global_model.save()
            session.reset_local_retrain_flag()
            print("Pushed update to GLOBAL. Global LSTM saved.")

        elif choice == "6":
            # Reuse server's admin menu (it will prompt for admin password)
            await server_app.admin_menu()

        elif choice == "7":
            print("Logging out.\n")
            break

        else:
            print("Invalid option.")

def show_scores(out, ytrue, label="GLOBAL"):
    p = out["p_fused"]
    y = ytrue
    if len(p) == 0:
        print("No data.")
        return
    # quick, friendly summary
    pred = (p >= 0.5).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    acc = (tp + tn) / max(1, len(y))
    print(f"[{label}] batch-size={len(y)} acc={acc:.3f} TP={tp} FP={fp} TN={tn} FN={fn}")
    # show top-5 most suspicious examples
    idx = np.argsort(-p)[:5]
    print("Top-5 suspicious scores:", [round(float(p[i]), 3) for i in idx])
