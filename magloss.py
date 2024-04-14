import os


def GetMagLoss(material_name, b_waveform, temp, freq):
    # save the current directory
    current_path = os.getcwd()

    path = "MagNet_package\MagNet_comb_" + material_name + "_cycle"
    os.chdir(path)

    import MagNet

    loss_pred = MagNet.MagLoss(b_waveform, temp, freq)

    # restore the current directory
    os.chdir(current_path)

    return loss_pred
