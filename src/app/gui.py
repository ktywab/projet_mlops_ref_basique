import tkinter as tk
from tkinter import ttk, messagebox

from predict import predict_texts

# Mets ici tes vrais noms si tu veux (ex: negatif/neutre/positif)
LABEL_MAP = {
    0: "positif",
    1: "neutre",
    2: "negatif",
}


def split_inputs(text: str):
    lines = [l.strip() for l in text.splitlines()]
    return [l for l in lines if l]


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sentiment Classifier (Offline)")
        self.geometry("950x540")

        top = tk.Frame(self)
        top.pack(fill="x", padx=10, pady=10)

        tk.Label(top, text="Commentaires (1 par ligne) :").pack(anchor="w")

        self.txt = tk.Text(top, height=8)
        self.txt.pack(fill="x", pady=6)

        btns = tk.Frame(self)
        btns.pack(fill="x", padx=10)

        tk.Button(btns, text="Predict", command=self.on_predict).pack(side="left", padx=5)
        tk.Button(btns, text="Clear", command=self.on_clear).pack(side="left", padx=5)

        res = tk.Frame(self)
        res.pack(fill="both", expand=True, padx=10, pady=10)

        cols = ("comment", "label", "label_name", "confidence", "proba", "note")
        self.tree = ttk.Treeview(res, columns=cols, show="headings")

        self.tree.heading("comment", text="Commentaire")
        self.tree.heading("label", text="Label")
        self.tree.heading("label_name", text="Classe")
        self.tree.heading("confidence", text="Score (max proba)")
        self.tree.heading("proba", text="Probabilités")
        self.tree.heading("note", text="Note")

        self.tree.column("comment", width=420)
        self.tree.column("label", width=60, anchor="center")
        self.tree.column("label_name", width=120, anchor="center")
        self.tree.column("confidence", width=140, anchor="center")
        self.tree.column("proba", width=250)
        self.tree.column("note", width=160, anchor="center")

        self.tree.pack(fill="both", expand=True, side="left")

        sb = ttk.Scrollbar(res, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscroll=sb.set)
        sb.pack(side="right", fill="y")

    def on_clear(self):
        self.txt.delete("1.0", tk.END)
        for item in self.tree.get_children():
            self.tree.delete(item)

    def on_predict(self):
        raw = self.txt.get("1.0", tk.END).strip()
        if not raw:
            messagebox.showwarning("Info", "Entre au moins un commentaire.")
            return

        comments = split_inputs(raw)
        if not comments:
            messagebox.showwarning("Info", "Aucun commentaire valide.")
            return

        for item in self.tree.get_children():
            self.tree.delete(item)

        try:
            results = predict_texts(comments, min_words=3)
        except Exception as e:
            messagebox.showerror("Erreur", str(e))
            return

        for comment, res in zip(comments, results):
            label = res["label"]
            note = res.get("note", "")
            if label is None:
                label_name = "-"
                conf_str = "-"
                proba_str = "-"
                label_val = "-"
            else:
                label_val = str(label)
                label_name = LABEL_MAP.get(label, f"Classe {label}")
                conf = res["confidence"]
                proba = res["proba"]
                conf_str = "" if conf is None else f"{conf:.4f}"
                proba_str = "" if proba is None else "[" + ", ".join(f"{p:.3f}" for p in proba) + "]"

            self.tree.insert("", tk.END, values=(comment, label_val, label_name, conf_str, proba_str, note))


if __name__ == "__main__":
    App().mainloop()