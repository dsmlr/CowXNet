def get_new_coordinates(x, y, xmin, ymin, xmax, ymax, revert=False):

    bbUpperLeftX = xmin
    bbUpperLeftY = ymax
    bbLowerRightX = xmax
    bbLowerRightY = ymin

    sizeX = bbLowerRightX - bbUpperLeftX # box width
    sizeY =  bbUpperLeftY - bbLowerRightY # box height

    centerX = (bbLowerRightX + bbUpperLeftX)/2
    centerY = (bbLowerRightY + bbUpperLeftY)/2

    offsetX = (centerX-sizeX/2) * sizeX/sizeX
    offsetY = (centerY-sizeY/2) * sizeY/sizeY
    
    if revert:
        w = xmax - xmin
        h = ymax - ymin

        dist_from_left = w - x
        dist_from_top = h - y

        x = xmax - dist_from_left
        y = ymax - dist_from_top

    else:

        x = (x * (sizeX/sizeX)) - offsetX 
        y = (y * (sizeY/sizeY)) - offsetY

    x = max(0, x)
    y = max(0, y)

    return (x, y)