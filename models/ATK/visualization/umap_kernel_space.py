import os

import numpy as np
import matplotlib.pyplot as plt
import umap
from mpl_toolkits.mplot3d import Axes3D 
from scipy.spatial import ConvexHull




def plot_umap_from_train_test_kernel(train_kernel,train_test_kernel ,test_kernel, Y_train, Y_test,
                                     save_path=None, fold_idx=0, n_neighbors=5, min_dist=0.1,
                                     random_state=42, normalize_kernel_fn=None, plot_3d=False):
    """
    Fit UMAP once trên full distance (train+test), rồi visualise embedding cho:
    1) Train only
    2) Test only
    3) Train + Test
    """
    def plot_embedding(embedding, labels, split, title, save_name=None):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d') if plot_3d else plt.gca()

        unique_classes = np.unique(labels)
        color_mapping = {0: 'orange', 1: 'blue'}
        markers = {'train': 'o', 'test': 'x'}

        if split == 'train':
            data_split = np.array(['train'] * len(labels))
        elif split == 'test':
            data_split = np.array(['test'] * len(labels))
        else:  # full
            data_split = np.array(['train'] * len(Y_train) + ['test'] * len(Y_test))

        for cls in unique_classes:
            color = color_mapping.get(cls, 'gray')
            for sp in ['train', 'test']:
                idx = (labels == cls) & (data_split == sp)
                if not np.any(idx): continue
                if plot_3d:
                    ax.scatter(embedding[idx, 0], embedding[idx, 1], embedding[idx, 2],
                               marker=markers[sp], label=f'Class {cls} ({sp})',
                               alpha=0.7, color=color)
                else:
                    ax.scatter(embedding[idx, 0], embedding[idx, 1],
                               marker=markers[sp], label=f'Class {cls} ({sp})',
                               alpha=0.7, color=color)

        if not plot_3d:
            for cls in unique_classes:
                idx = labels == cls
                if np.sum(idx) >= 3:
                    pts = embedding[idx]
                    try:
                        hull = ConvexHull(pts)
                        for simplex in hull.simplices:
                            plt.plot(pts[simplex, 0], pts[simplex, 1], 'k-', linewidth=0.5)
                    except:
                        pass

        if plot_3d:
            ax.set_xlabel('UMAP1'); ax.set_ylabel('UMAP2'); ax.set_zlabel('UMAP3')
        else:
            plt.xlabel('UMAP1'); plt.ylabel('UMAP2')

        plt.title(title)
        plt.legend(title='Labels + Split', loc='best', fontsize=8)
        plt.tight_layout()

        if save_name:
            os.makedirs(os.path.dirname(save_name), exist_ok=True)
            plt.savefig(save_name, dpi=300)
            plt.close()
        else:
            plt.show()

    def generate_distance_matrix(kernel):
        if normalize_kernel_fn is not None:
            kernel = normalize_kernel_fn(kernel)
        if kernel.shape[0] == kernel.shape[1]:
            diag = np.diag(kernel)
        else:
            diag = np.sum(kernel**2, axis=1)
        return np.sqrt(np.maximum(diag[:, None] + diag[None, :] - 2 * kernel, 0))

    # --- Tạo full kernel và labels ---
    n_train = train_kernel.shape[0]
    # full_kernel = [[K_tt, K_tT],
    #                [K_Tt, K_TT]]
    top    = np.concatenate([train_kernel, train_test_kernel], axis=1)
    bottom = np.concatenate([train_test_kernel.T, test_kernel], axis=1)
    full_kernel = np.concatenate([top, bottom], axis=0)
    full_labels = np.concatenate([Y_train, Y_test], axis=0)

    # --- Fit UMAP một lần trên full distance ---
    full_distance = generate_distance_matrix(full_kernel)
    reducer = umap.UMAP(
        n_neighbors=n_neighbors, min_dist=min_dist, metric='precomputed',
        random_state=random_state, n_components=3 if plot_3d else 2
    )
    embedding_full = reducer.fit_transform(full_distance)

    # 1) Vẽ train chỉ với phần đầu của embedding_full
    plot_embedding(
        embedding_full[:n_train], Y_train, split='train',
        title='UMAP Projection – Train Only',
        save_name=os.path.join(save_path, f'umap_train_fold{fold_idx}.png') if save_path else None
    )

    # 2) Vẽ test với phần còn lại
    plot_embedding(
        embedding_full[n_train:], Y_test, split='test',
        title='UMAP Projection – Test Only',
        save_name=os.path.join(save_path, f'umap_test_fold{fold_idx}.png') if save_path else None
    )

    # 3) Vẽ full với toàn bộ embedding
    plot_embedding(
        embedding_full, full_labels, split='full',
        title='UMAP Projection – Train + Test',
        save_name=os.path.join(save_path, f'umap_full_fold{fold_idx}.png') if save_path else None
    )
