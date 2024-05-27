"""
Reading Order
"""


def updateRO_old(page, RegROgroups):
    indices = [i for i in range(len(page))]
    for grp in RegROgroups:
        selected_reg = [page[i] for i in indices if page[i]['type'] in grp]
        selected_boxes = [boxes[i] for i in indices if labels[i] in grp]
        selected_labels = [labels[i] for i in indices if labels[i] in grp]

        sorted_masks = [x for x, _, _ in
                        sorted(zip(selected_masks, selected_boxes, selected_labels), key=lambda pair: pair[1][0][1])]
        sorted_boxes = [x for _, x, _ in
                        sorted(zip(selected_masks, selected_boxes, selected_labels), key=lambda pair: pair[1][0][1])]
        sorted_labels = [x for _, _, x in
                         sorted(zip(selected_masks, selected_boxes, selected_labels), key=lambda pair: pair[1][0][1])]

        for x, y, z in zip(sorted_masks, sorted_boxes, sorted_labels):
            n_masks.append(x)
            n_boxes.append(y)
            n_labels.append(z)
    return n_masks, n_boxes, n_labels


def updateRegionRO(opts, page, ROgroups):
    n_regs = []
    indices = [i for i in range(len(page))]
    for grp in ROgroups:
        selected_regs = [page[i] for i in indices if page[i]['type'] in grp]
        if opts.DEBUG: print(selected_regs)
        sorted_regs = [x for x in sorted(selected_regs, key=lambda pair: pair['bbox'][0][1])]

        for x in sorted_regs:
            n_regs.append(x)

    return n_regs


def updateLineRO(opts, lines):
    n_lines = []
    # indices = [i for i in range(len(lines))]
    if opts.DEBUG: print(lines)
    sorted_lines = [x for x in sorted(lines, key=lambda line: line['bbox'][0][1])]

    for x in sorted_lines:
        n_lines.append(x)

    return n_lines