from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
import numpy as np

multiclass = np.array([[13,0,0,3,0,2,2],
                       [0,20,0,0,0,0,0],
                       [0,0,20,0,0,0,0],
                       [0,0,0,20,0,0,0],
                       [0,0,0,0,20,0,0],
                       [2,0,0,3,0,15,0],
                       [0,0,0,0,0,0,20]
                    ])

class_names = ["A", "B", "C", "D", "E", "F", "G"]

# fig, ax = plot_confusion_matrix(conf_mat=binary1)
# fig, ax = plot_confusion_matrix(conf_mat=binary,
#                                 show_absolute=True,
#                                 show_normed=True,
#                                 colorbar=True,
#                                 figsize=(15,15))
fig, ax = plot_confusion_matrix(conf_mat=multiclass,
                                # colorbar=True,
                                # show_absolute=False,
                                show_absolute=False,
                                show_normed=True,
                                # class_names=class_names,
                                figsize=(15, 15)
                                )
# plt.show()
plt.savefig('confusion_matrix.png')
