import matplotlib.pyplot as plt

def show_plot(x_data, y_data, y_model):
    fig, ax = plt.subplots()
    ax.grid(True, which="major", axis="both")
    ax.plot(x_data, y_data, label="Data")
    ax.plot(x_data, y_model, label="Model")
    plt.legend()
    plt.show()

