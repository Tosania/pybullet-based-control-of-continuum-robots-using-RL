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
                 reward_fn, done_fn,
                 log_dir="./ppo_log/PPO_2E7", output_path="report.pdf"):

        self.model_name = model_name
        self.total_timesteps = total_timesteps
        self.control_mode = control_mode
        self.device = device
        self.net_arch = net_arch
        self.reward_fn = reward_fn
        self.done_fn = done_fn
        self.output_path = output_path
        self.log_dir = log_dir
        self.judge = judge
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    def plot_tensorboard_scalar(self, tag="rollout/ep_rew_mean", output_path="reward_curve.png"):
        ea = event_accumulator.EventAccumulator(self.log_dir)
        ea.Reload()
        print("📋 可用的 Tags:")
        print(ea.Tags()["scalars"])
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
            ("Timesteps", f"{self.total_timesteps}"),
            ("Control Mode", str(self.control_mode)),
            ("Device", self.device),
            ("Network Arch", str(self.net_arch)),
            ("Mean Error", self.judge)
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

        image_path = self.plot_tensorboard_scalar()
        if image_path and os.path.exists(image_path):
            pdf.image(image_path, w=180)
        else:
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 10, txt="can't load reward", ln=True)

        pdf.output(self.output_path)
        print(f"PDF报告已生成: {self.output_path}")