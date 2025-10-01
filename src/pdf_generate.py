from fpdf import FPDF
from datetime import datetime
import textwrap
import inspect
import os
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
os.environ["QT_QPA_PLATFORM"] = "offscreen"
class ReportGenerator:
    def __init__(self, model_name, total_timesteps, judge,
                 control_mode, device, net_arch,
                 reward_fn, done_fn,n_env,seed,
                 learning_rate, batch_size, n_steps,n_epochs,best_reward,
                 buffer_size,
                 learning_starts,
                 train_freq,
                 model_type,
                 log_dir="./ppo_log/PPO_2E7", output_path="report.pdf"):

        self.model_name = model_name
        self.total_timesteps = total_timesteps
        self.control_mode = control_mode
        self.device = device
        self.net_arch = net_arch
        self.reward_fn = reward_fn
        self.seed=seed
        self.learning_rate=learning_rate
        self.batch_size=batch_size
        self.n_steps=n_steps
        self.best_reward=best_reward
        self.buffer_size=buffer_size
        self.train_freq=train_freq
        self.learning_starts=learning_starts
        self.model_type=model_type
        self.n_env=n_env
        self.n_epochs=n_epochs
        self.done_fn = done_fn
        self.num=0
        self.output_path = output_path
        self.log_dir = log_dir
        self.judge = judge
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    def plot_tensorboard_scalar(self, tag, output_path="./temp/reward_curve.png"):
        output_path=f"./temp/reward_curve_{self.num}.png"
        self.num+=1
        ea = event_accumulator.EventAccumulator(self.log_dir)
        ea.Reload()
        if tag not in ea.Tags()["scalars"]:
            print(f"⚠️ Tag '{tag}' not found in logs.")
            return None

        events = ea.Scalars(tag)
        steps = [e.step for e in events]
        values = [e.value for e in events]

        plt.figure(figsize=(6, 4))
        plt.plot(steps, values, label=tag)
        plt.xlabel("Steps")
        plt.ylabel("Value")
        plt.title(f"TensorBoard: {tag}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        return output_path

    def generate_pdf(self):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.set_title("Continuum Robot PPO report")

        pdf.set_font("Arial", 'B', size=16)
        pdf.cell(200, 10, txt="Continuum Robot PPO report", ln=True, align='C')
        pdf.ln(10)

        pdf.set_font("Arial", size=12)
        rows = [
            ("Model Name", self.model_name),
            ("Time", self.timestamp),
            ("Model_Type",self.model_type),
            ("seed",self.seed),
            ("Timesteps", f"{self.total_timesteps}"),
            ("Control Mode", str(self.control_mode)),
            ("Device", self.device),
            ("Network Arch", str(self.net_arch)),
            ("Average Error", self.judge),
            ("batch",self.batch_size),
            ("buffer_size",self.buffer_size),
            ("train_freq",self.train_freq),
            ("learning_starts",self.learning_starts),
            ("n_steps",self.n_steps),
            ("n_epochs",self.n_epochs),
            ("learning_rate",str(self.learning_rate)),
            ("n_env",self.n_env),
            ("best_reward",str(self.best_reward))
        ]

        for name, value in rows:
            pdf.cell(60, 10, txt=name, border=1)
            pdf.cell(130, 10, txt=str(value), border=1)
            pdf.ln()

        # 输出 reward 函数源码
        pdf.ln(5)
        pdf.set_font("Arial", 'B', size=14)
        pdf.cell(200, 10, txt="Reward Function Source", ln=True)

        pdf.set_font("Courier", size=10)
        try:
            code = inspect.getsource(self.reward_fn)
        except TypeError:
            code = "# Unable to extract reward_fn source"

        lines = code.splitlines()
        if len(lines) == 1:
            lines = textwrap.wrap(lines[0], width=60)

        for line in lines:
            pdf.multi_cell(0, 6, txt=line)

        # 输出 done 函数源码
        pdf.ln(5)
        pdf.set_font("Arial", 'B', size=14)
        pdf.cell(200, 10, txt="Done Function Source", ln=True)

        pdf.set_font("Courier", size=10)
        try:
            code = inspect.getsource(self.done_fn)
        except TypeError:
            code = "# Unable to extract done_fn source"

        lines = code.splitlines()
        if len(lines) == 1:
            lines = textwrap.wrap(lines[0], width=60)

        for line in lines:
            pdf.multi_cell(0, 6, txt=line)

        # 加入 TensorBoard 曲线图
        pdf.ln(5)
        pdf.set_font("Arial", 'B', size=14)
        pdf.cell(200, 10, txt="Training Reward Curve", ln=True)
        ea = event_accumulator.EventAccumulator(self.log_dir)
        ea.Reload()
        for i in ea.Tags()["scalars"]:
          print(i)
          image_path = self.plot_tensorboard_scalar(tag=i)
          if image_path and os.path.exists(image_path):
              pdf.image(image_path, w=180)
          else:
              pdf.set_font("Arial", size=12)
              pdf.cell(200, 10, txt="can't load reward", ln=True)
        pdf.output(self.output_path)
        print(f"PDF报告已生成: {self.output_path}")