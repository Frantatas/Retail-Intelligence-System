import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import joblib

# -----------------------------
# Load trained models
# -----------------------------
lr_model = joblib.load("lr_model.pkl")
log_model = joblib.load("log_model.pkl")
dt_model = joblib.load("dt_model.pkl")
rf_model = joblib.load("rf_model.pkl")
nn_model = joblib.load("nn_model.pkl")
scaler = joblib.load("scaler.pkl")
feature_columns = joblib.load("feature_columns.pkl")

# -----------------------------
# Predict function
# -----------------------------
def predict_all():
    try:
        quantity = int(entry_qty.get())
        unit_price = float(entry_price.get())
        month = int(entry_month.get())

        if month < 1 or month > 12:
            messagebox.showerror("Input Error", "Month must be between 1 and 12.")
            return

        customer = combo_cust.get()
        gender = combo_gender.get()
        category = combo_cat.get()
        payment = combo_pay.get()

        input_data = pd.DataFrame([{
            "Quantity": quantity,
            "Unit price": unit_price,
            "Month": month,
            "Customer type": customer,
            "Gender": gender,
            "Product line": category,
            "Payment": payment
        }])

        input_encoded = pd.get_dummies(input_data)
        input_encoded = input_encoded.reindex(columns=feature_columns, fill_value=0)

        # Keep numeric values as float/int, only convert dummy bools if needed
        for col in input_encoded.columns:
            if input_encoded[col].dtype == bool:
                input_encoded[col] = input_encoded[col].astype(int)

        lr_pred = lr_model.predict(input_encoded)[0]
        log_pred = log_model.predict(input_encoded)[0]
        dt_pred = dt_model.predict(input_encoded)[0]
        rf_pred = rf_model.predict(input_encoded)[0]

        input_scaled = scaler.transform(input_encoded)
        nn_pred = nn_model.predict(input_scaled)[0]

        log_prob = log_model.predict_proba(input_encoded)[0][1]
        dt_prob = dt_model.predict_proba(input_encoded)[0][1]
        rf_prob = rf_model.predict_proba(input_encoded)[0][1]
        nn_prob = nn_model.predict_proba(input_scaled)[0][1]

        lbl_lr.config(text=f"PHP {lr_pred:,.2f}")
        lbl_log.config(text=f"{'HIGH Sale' if log_pred == 1 else 'LOW Sale'} ({log_prob*100:.1f}%)")
        lbl_dt.config(text=f"{'HIGH Sale' if dt_pred == 1 else 'LOW Sale'} ({dt_prob*100:.1f}%)")
        lbl_rf.config(text=f"{'HIGH Sale' if rf_pred == 1 else 'LOW Sale'} ({rf_prob*100:.1f}%)")
        lbl_nn.config(text=f"{'HIGH Sale' if nn_pred == 1 else 'LOW Sale'} ({nn_prob*100:.1f}%)")

    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values.")
    except Exception as e:
        messagebox.showerror("Error", str(e))


# -----------------------------
# Window
# -----------------------------
root = tk.Tk()
root.title("Retail Intelligence System")
root.geometry("820x680")
root.configure(bg="#1A1A1D")
root.resizable(False, False)

style = ttk.Style()
style.theme_use("clam")

style.configure(
    "Custom.TCombobox",
    fieldbackground="#E7D6E7",
    background="#E7D6E7",
    foreground="#1A1A1A",
    borderwidth=0,
    padding=6
)

# -----------------------------
# Header
# -----------------------------
tk.Label(
    root,
    text="Retail Intelligence System",
    font=("Segoe UI", 20, "bold"),
    fg="#F5F5F5",
    bg="#1A1A1D"
).pack(pady=(16, 4))

tk.Label(
    root,
    text="Enter supermarket transaction details to get ML predictions",
    font=("Segoe UI", 10),
    fg="#B9B6C9",
    bg="#1A1A1D"
).pack()

# -----------------------------
# Input card
# -----------------------------
frame_in = tk.Frame(root, bg="#242428", bd=0)
frame_in.pack(fill="x", padx=20, pady=14)

product_options = [
    "Health and beauty",
    "Electronic accessories",
    "Home and lifestyle",
    "Sports and travel",
    "Food and beverages",
    "Fashion accessories"
]

payment_options = ["Cash", "Credit card", "Ewallet"]

def add_field(parent, label, row, col, widget):
    tk.Label(
        parent,
        text=label,
        font=("Segoe UI", 10, "bold"),
        bg="#242428",
        fg="#F2F2F2"
    ).grid(row=row, column=col, sticky="w", padx=14, pady=8)
    widget.grid(row=row, column=col+1, padx=14, pady=8, sticky="ew")

def make_entry(parent):
    return tk.Entry(
        parent,
        font=("Segoe UI", 10),
        bg="#D9CFE3",
        fg="#1A1A1A",
        relief="flat",
        insertbackground="#1A1A1A",
        width=18
    )

entry_qty = make_entry(frame_in)
entry_price = make_entry(frame_in)
entry_month = make_entry(frame_in)

combo_cust = ttk.Combobox(frame_in, values=["Member", "Normal"], width=18, state="readonly", style="Custom.TCombobox")
combo_gender = ttk.Combobox(frame_in, values=["Male", "Female"], width=18, state="readonly", style="Custom.TCombobox")
combo_cat = ttk.Combobox(frame_in, values=product_options, width=18, state="readonly", style="Custom.TCombobox")
combo_pay = ttk.Combobox(frame_in, values=payment_options, width=18, state="readonly", style="Custom.TCombobox")

entry_qty.insert(0, "7")
entry_price.insert(0, "74.69")
entry_month.insert(0, "1")

combo_cust.set("Member")
combo_gender.set("Female")
combo_cat.set("Food and beverages")
combo_pay.set("Ewallet")

add_field(frame_in, "Quantity:", 0, 0, entry_qty)
add_field(frame_in, "Unit Price:", 1, 0, entry_price)
add_field(frame_in, "Month (1-12):", 2, 0, entry_month)
add_field(frame_in, "Customer Type:", 0, 2, combo_cust)
add_field(frame_in, "Gender:", 1, 2, combo_gender)
add_field(frame_in, "Product Line:", 2, 2, combo_cat)
add_field(frame_in, "Payment:", 3, 2, combo_pay)

# -----------------------------
# Predict button
# -----------------------------
tk.Button(
    root,
    text="Get Predictions from All 5 Models",
    command=predict_all,
    font=("Segoe UI", 11, "bold"),
    bg="#F3E7EF",
    fg="#111111",
    activebackground="#E7D6E7",
    activeforeground="#111111",
    padx=22,
    pady=10,
    relief="flat",
    cursor="hand2"
).pack(pady=12)

# -----------------------------
# Results card
# -----------------------------
frame_out = tk.Frame(root, bg="#242428")
frame_out.pack(fill="x", padx=20, pady=(6, 18))

tk.Label(
    frame_out,
    text="Model Predictions",
    font=("Segoe UI", 13, "bold"),
    bg="#242428",
    fg="#F5F5F5"
).grid(row=0, column=0, columnspan=2, pady=12)

models = [
    ("Linear Regression", "lbl_lr", "#CFE0EF"),
    ("Logistic Regression", "lbl_log", "#E8D4E3"),
    ("Decision Tree", "lbl_dt", "#DAD2F0"),
    ("Random Forest", "lbl_rf", "#DDE8D3"),
    ("Neural Network", "lbl_nn", "#F3E3CC")
]

for i, (model_name, var_name, color) in enumerate(models):
    row_frame = tk.Frame(frame_out, bg=color)
    row_frame.grid(row=i+1, column=0, columnspan=2, sticky="ew", padx=18, pady=6)

    tk.Label(
        row_frame,
        text=model_name,
        font=("Segoe UI", 10, "bold"),
        bg=color,
        fg="#1A1A1A",
        width=18,
        anchor="w"
    ).pack(side="left", padx=12, pady=10)

    lbl = tk.Label(
        row_frame,
        text="—",
        font=("Segoe UI", 10),
        bg=color,
        fg="#1A1A1A",
        width=28,
        anchor="w"
    )
    lbl.pack(side="left", padx=8, pady=10)

    globals()[var_name] = lbl

root.mainloop()