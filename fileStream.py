def initFile(filename):
    file_object = open(filename, "a")
    file_object.write("Epoch\tAUC\tSpec\tSens\tval_loss\n")
    file_object.close()


def writeToFile(filename, epoch, auc, spec, sens, val_loss):
    file_object = open(filename, "a")
    file_object.write("%d\t%f\t%f\t%f\t%f\n" % (epoch, auc, spec, sens, val_loss))
    file_object.close()