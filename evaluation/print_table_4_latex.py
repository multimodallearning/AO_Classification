from evaluation.quantitative import mean_df, pd, mode


def print_mark(key, name):
    if key in name:
        return '\checkmark'
    else:
        return ''


if mode == 'end2end':
    table = pd.DataFrame(columns=['Img', 'Loc', 'Seg', 'Rep', 'Accuracy', 'F1', 'Precision', 'Recall', 'AUROC'])
    for experiment, row in mean_df.iterrows():
        table = pd.concat([table, pd.DataFrame({
            'Img': print_mark('image', experiment),
            'Loc': print_mark('loc', experiment),
            'Seg': print_mark('seg', experiment),
            'Rep': print_mark('clip', experiment),
            'Accuracy': row['Accuracy'],
            'F1': row['F1'],
            'Precision': row['Precision'],
            'Recall': row['Recall'],
            'AUROC': row['AUROC']
        }, index=[0]), ], ignore_index=True)
    # multiply floats by 100 to get percentage
    table.iloc[:, 4:] *= 100
    table = table.round(2)

    print('\n\n')
    print(table.to_latex(index=False, float_format='%.2f'))
else:
    table = pd.DataFrame(columns=['Encoder', 'Accuracy', 'F1', 'AUROC'])
    for experiment, row in mean_df.iterrows():
        table = pd.concat([table, pd.DataFrame({
            'Encoder': experiment,
            'Accuracy': row['Accuracy'],
            'F1': row['F1'],
            'AUROC': row['AUROC']
        }, index=[0]), ], ignore_index=True)
    # multiply floats by 100 to get percentage
    table.iloc[:, 1:] *= 100
    table = table.round(2)

    print('\n\n')
    print(table.to_latex(index=False, float_format='%.2f'))
