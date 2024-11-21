# In plotting.py
def plot_progress(average_loss_per_gen, survival_counts, average_ethical_score_per_gen, filename='progress_plot.png'):
    fig, ax1 = plt.subplots(figsize=(12, 8))

    ax1.plot(range(1, len(average_loss_per_gen) + 1), average_loss_per_gen, label='Average Loss per Generation', marker='o')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Loss', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.plot(range(1, len(survival_counts) + 1), survival_counts, label='Rounds Survived per Game', marker='x', color='orange')

    ax2 = ax1.twinx()
    ax2.plot(range(1, len(average_ethical_score_per_gen) + 1), average_ethical_score_per_gen, label='Average Ethical Score per Generation', marker='^', color='green')
    ax2.set_ylabel('Ethical Score', color='green')
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='y', labelcolor='green')

    fig.legend(loc="upper right", bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes)
    plt.title('Progress of Survival, Loss, and Ethical Scores Across Generations')
    plt.savefig(filename)
    plt.close()
    print(f"Plot saved as '{filename}'")

