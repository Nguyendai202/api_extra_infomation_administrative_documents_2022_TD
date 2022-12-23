def splitRowsInDocument(img_path):
  #dfff
    points = []
    other_points = []
    img0 = cv2.imread(img_path)
    img = img0.copy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    thresh_image = cv2.threshold(img, 175, 255, cv2.THRESH_BINARY)[1]
    height = img.shape[0]
    width = img.shape[1]  
    image_list = []
    value_y = []
    index_y = []
    split_y_const = 5
    split_index_y = []
    for h in range(0, height):
        row = []
        sum = 0
        for w in range(0, width):
            row.append(thresh_image[h][w])
            sum += thresh_image[h][w]
        value = float(sum/len(row))
        value_y.append([h, value])
  ###############
    for y in value_y:
        if y[1] < 253.0:
            top = y[0]
            break
    back_indx = len(value_y)
    while(back_indx >= 0):
        if value_y[back_indx - 1][1] <253.0:
            bottom = value_y[back_indx - 1][0]
            break
        else: 
            back_indx -= 1
    for y in value_y:
        if y[1] >= 245.0:
            index_y.append(y[0])
    for i in range(0, len(index_y) - 1):
        if index_y[i + 1] - index_y[i] >= split_y_const:
            split_index_y.append([index_y[i], index_y[i + 1]])
  ###########################
    if len(split_index_y) > 1:
        steps = []
        for i in range(1, len(split_index_y)):
            value = int((split_index_y[i - 1][1] + split_index_y[i][0]) / 2)
            steps.append(value)
        left = split_index_y[0][0]
    ###########################
        points.append(top)
        for i in range(0, len(steps)):
            points.append(steps[i])
        points.append(bottom)
        for i in range(0, len(points) - 2):
            rate = (points[i + 2] - points[i + 1]) / (points[i + 1] - points[i])
            if rate >= 1.6:
                points[i + 1] = -1
        for point in points:
            if point >= 0:
                other_points.append(point)
        for i in range(0, len(other_points) - 1):
            image_list.append(img0[other_points[i] : other_points[i + 1]])
    else:
        image_list.append(img0)
    return image_list