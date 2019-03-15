import cv2
import json
import os

def draw_bbox( input_img, bbox, label_name, color = (0, 255, 0) ):
    """
    draw a bounding box with transparence rate
    :param input_img: OpenCV cv2 read image
    :param bbox: bbox is a list with [ w_min, h_min, w_max, h_max ]
    :param label_name: label is a string indicates the label name
    :param color: a tuple indicating the color to draw the bounding box
    :return: an image that is savable
    """
    overlay = input_img.copy()
    output_img = input_img.copy()

    w_min = bbox[0]
    h_min = bbox[1]
    w_max = bbox[2]
    h_max = bbox[3]

    font_scale = 0.5
    thickness = 1

    text_size = cv2.getTextSize( label_name, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness )
    #print( 'text_size = ', text_size )
    #print( 'text_size = ', text_size[0][0] )
    text_width = int( text_size[0][0] )
    text_height = int( text_size[0][1] )


    img_h, img_w, img_c = input_img.shape
    h_begin = h_min - ( text_height + 5 )
    text_h_begin = h_min - 5
    h_end = h_min
    if h_begin < 0:
        h_begin = h_min
        h_end = h_begin + text_height + 5
        text_h_begin = h_min + 15

    w_end = w_min + ( text_width + 5 )
    if w_end > img_w:
        w_end = img_w

    transparent_bbox = [ w_min, h_begin , w_end, h_end ]
    cv2.rectangle( overlay, ( transparent_bbox[0], transparent_bbox[1] ),
                            ( transparent_bbox[2], transparent_bbox[3] ),
                            color,
                            -1 )
    alpha = 0.5
    cv2.addWeighted( overlay, alpha, output_img, 1 - alpha, 0, output_img )
    cv2.rectangle( output_img, ( w_min, h_min ), ( w_max, h_max ), color, 2 )

    cv2.putText( output_img, label_name, ( w_min, text_h_begin ), cv2.FONT_HERSHEY_SIMPLEX,
                 0.5, ( 255, 255, 255 ), 1 )

    return output_img



class_list_file = '/home/yuhang/project_huawei_smartphone/huawei_102cate_test_0131/102cate_class_list.txt'
class_dict = dict()

class_list = [ line_tmp.rstrip('\n') for line_tmp in open( class_list_file, 'r' ).readlines() ]
for line_tmp in class_list:
    label_id = line_tmp.split(' ')[0]
    label_name = ' '.join( line_tmp.split(' ')[1:] )
    class_dict[ label_id ] = label_name

gt_file_name = '/home/yuhang/project_huawei_smartphone/huawei_102cate_test_0131/fpn_320_result.txt'
gt_list = [ line_tmp.rstrip('\n') for line_tmp in open( gt_file_name, 'r' ).readlines() ]
img_source_dir = '/opt/FTE/users/opendata/mscoco/val2017'
img_save_dir = '/home/yuhang/project_huawei_smartphone/huawei_102cate_test_0131/pre_bbox'

for idx, line_tmp in enumerate( gt_list ):
    if idx%10 == 0 and idx > 0:
        break
    [img_base_name, cate_id, xmin, ymin, xmax, ymax, score ] = line_tmp.split(',')
    img_full_name = os.path.join( img_source_dir, img_base_name )
    img_tmp = cv2.imread( img_full_name, 1 )
    if img_tmp is None:
        print('cannot read the image {}'.format( img_full_name ) )
        continue
    xmin = int( float( xmin ) )
    ymin = int(float(ymin))
    xmax = int(float(xmax))
    ymax = int(float(ymax))

    img_tmp = draw_bbox( img_tmp, [xmin, ymin, xmax, ymax], class_dict[ cate_id ] )

    img_save_name = os.path.join( img_save_dir, str(idx) + '_' + img_base_name )

    cv2.imwrite( img_save_name, img_tmp )

print('Done!')
