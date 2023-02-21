import tkinter as tk
import random

class StroopTask(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.master = master
        self.colors = ["red", "green", "blue", "yellow"]
        self.color_words = ["RED", "GREEN", "BLUE", "YELLOW"]
        self.score = 0
        self.total_trials = 0
        self.create_widgets()

    def create_widgets(self):
        self.color_label = tk.Label(self.master, font=("Arial", 40), pady=20)
        self.color_label.pack()
        self.user_entry = tk.Entry(self.master, font=("Arial", 20), width=10)
        self.user_entry.pack(pady=10)
        self.score_label = tk.Label(self.master, text="Score: 0", font=("Arial", 20))
        self.score_label.pack()
        self.start_button = tk.Button(self.master, text="Start", font=("Arial", 20), command=self.start_task)
        self.start_button.pack(pady=10)

    def start_task(self):
        self.start_button.config(state="disabled")
        self.user_entry.config(state="normal")
        self.total_trials += 1
        color_word = random.choice(self.color_words)
        color = random.choice(self.colors)
        self.color_label.config(text=color_word, fg=color)
        self.user_entry.delete(0, tk.END)
        self.master.after(2000, self.check_answer)

    def check_answer(self):
        user_input = self.user_entry.get()
        if user_input.upper() == self.color_words[self.colors.index(self.color_label["fg"])]:
            self.score += 1
        self.score_label.config(text=f"Score: {self.score}")
        self.start_button.config(state="normal")
        self.user_entry.config(state="disabled")

root = tk.Tk()
app = StroopTask(root)
app.pack()
root.mainloop()
