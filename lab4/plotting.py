
import matplotlib.pyplot as plt

def cost_plot(opt, destfile):
    plt.plot(opt.epoch_nums, opt.cost_train, 'r-', label='Train')
    plt.plot(opt.epoch_nums, opt.cost_val, 'b-', label='Validation')
    plt.title('Smoothed cost function')
    plt.xlabel('Sequences')
    plt.ylabel('Cost')
    plt.legend()
    plt.grid('on')

    plt.savefig('images/' + destfile)
    plt.clf()
    plt.close()
    return 'images/' + destfile
