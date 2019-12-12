import PIL
import PIL
from PIL import Image
import numpy as np
from os import listdir
import glob
import matplotlib.pyplot as plt
import cv2
from scipy.signal import convolve2d

def Orig_to_GreySc():
    name_value = 1
    for filename in glob.glob('Iris_Training_Dataset_Orig/*.JPG'):
        im = Image.open(filename)
        gs_im = im.convert('L')
        gs_im.save('Iris_Training_Dataset_GS/Image_GS_{}.JPG'.format(str(name_value)))
        name_value += 1

def Qual_Reduction():
    ''' Reduces the quality of a picture by 50%. '''
    name_value = 1
    for filename in glob.glob('Iris_Training_Dataset_GS_Cropped/*.JPG'):
        im = Image.open(filename)
        im.save('Iris_Training_Dataset_GS_CrandRedQ/Image_GS_{}.JPG'.format(str(name_value)), quality=50)
        name_value += 1

def RGB_reduction(image):
    ''' Returns a matrix, each pixel represented by one number from the greyscale. '''
    # Checks if each picture is greyscaled first.
    # Makes a copy and opens in an array type.
    if not isinstance(image, np.ndarray):
        opened_img = cv2.imread(image)
        opened_img = np.asanyarray(opened_img)
    opened_img = image
    opened_img.setflags(write = 1)
    # Checking of greyscaled.
    if opened_img[0][0][0] == opened_img[0][0][1] and opened_img[0][0][0] == opened_img[0][0][2]:
        # If they are, deletes the first and second by index elements of each pixel, since each pixel has the same three values each.
        new_image_matrix = []
        row_numb = 0
        for row in opened_img:
            #pixel_numb = 0
            blank_row = []
            for pixel in opened_img[row_numb]:
                new_pixel = pixel[0]
                blank_row.append(new_pixel)
                #print(pixel)
                #row[pixel_numb] = pixel[0]
                #pixel_numb += 1
                #print(row)
            new_image_matrix.append(blank_row)
            row_numb += 1
    # If not greyscaled, raises a TypeError.
    else:
        raise TypeError("Pixels must be greyscaled first.")
    # Returns the final list.
    return new_image_matrix

def image_from_array_reconstruction(matrix):
    ''' Works for sure. Array into image. '''
    blank_matrix = []
    for row in matrix:
        row_count = 0
        blank_row = []
        for i in row:
            pixel = [i, i, i]
            blank_row.append(pixel)
        row_count += 1
        blank_matrix.append(blank_row)
    img = np.asarray(blank_matrix).astype('uint8')
    img = Image.fromarray(img)
    return img

#    Image.fromarray(data, 'RGB')

    '''w, h = len(inp_array), len(inp_array[0])
    data = np.zeros((h, w, 1), dtype=np.uint8)
    data[0:256, 0:256] = [255, 0, 0]  # red patch in upper left
    img = Image.fromarray(data, 'RGB')
    print(img)
    #img.save('my.png')
    #img.show()'''

def convolve(matrix_1, matrix_2):
    array_1 = np.array(matrix_1)
    array_2 = np.array(matrix_2)
    convolved_matrix = convolve2d(array_1, array_2, 'valid')
    return convolved_matrix


    '''dim_1 = (len(matrix_1[0]), len(matrix_1[0]))
    dim_2 = (len(matrix_2[0]), len(matrix_2[0]))
    blank_matrix = []
    row_counter = 0
    for by_row in range(len(matrix_1) - matrix_1[1]):
        blank_row = []
        column_counter = 0
        for by_column in range(len(matrix_1[0]) - matrix_1[0]):
            pixel =
            blank_row.append()
        row_counter += 1
        blank_matrix.append()'''

def convolution_layer(matrix, features_folder):
    features_list = []
    for item in glob.glob(features_folder + '/*.JPG'):
        feature_img = cv2.imread(item)
        feature_matrix = RGB_reduction(feature_img)
        features_list.append(feature_matrix)
        feature_matrix = image_from_array_reconstruction(feature_matrix)
        feature_matrix.save('feature_matrix1.JPG')
    output_list_of_matrices = []
    for feature_matrix in features_list:
        convolved_matrix = convolve(matrix, feature_matrix)
        output_list_of_matrices.append(convolved_matrix)
    return output_list_of_matrices

def pooling_layer(matrix, dimension):
    blank_matrix = []
    counter_1 = 0
    for i in range(int(len(matrix)/dimension)):
        blank_row = []
        counter_2 = 0
        for j in range(int(len(matrix[0])/dimension)):
            blank_row.append(int(np.mean(matrix[counter_1:(counter_1 + dimension), counter_2:(counter_2 + dimension)])))
            counter_2 += dimension
        blank_matrix.append(blank_row)
        counter_1 += dimension
    return blank_matrix

def vectorize_matrix(matrix):
    blank_vector = []
    matrix = np.asanyarray(matrix)
    for i in range(len(matrix[0])):
        blank_vector.extend(matrix[:, i])
    return blank_vector

def vectorize_multiple_matrices(a_list):
    blank_vector = []
    for matrix in a_list:
        micro_blank_vector = vectorize_matrix(matrix)
        blank_vector.extend(micro_blank_vector)
    return blank_vector

def vectors_in_vspace(training_dataset_address, identity_texts):
    vectors_in_vspace = []
    counter = 1
    for file in glob.glob(training_dataset_address + '/Image_GS_*.JPG'):
        print(training_dataset_address + '/Image_GS_{}.JPG'.format(counter))
        image_1 = cv2.imread(training_dataset_address + '/Image_GS_{}.JPG'.format(counter))
        image_1_array = np.asarray(image_1)
        image_1_RGB_reduced = RGB_reduction(image_1_array)
        image_1_convolved = convolution_layer(image_1_RGB_reduced, features_folder='Features_for_Convolution')
        pooled_list = []
        for item in image_1_convolved:
            pooled_list.append(pooling_layer(item, 13))
        image_1_vector = vectorize_multiple_matrices(pooled_list)
        vectors_in_vspace.append(image_1_vector)
        print("iteration " + str(counter))
        counter += 1
    vectors_in_vspace_id = []
    id_numb = 0
    for vector in vectors_in_vspace:
        vectors_in_vspace_id.append((id_numb, vector, identity_texts[id_numb]))
        id_numb += 1
    vectors_in_vspace_id.sort()
    return vectors_in_vspace_id

def find_dist(v1, v2):
    v1_t_a = np.array(tuple(v1))
    v2_t_a = np.array(tuple(v2))
    return np.linalg.norm(v1_t_a-v2_t_a)

def find_match(vectors_in_vspace_id, some_vector):
    distance_list = []
    navig_dict = {}
    for i in range(len(vectors_in_vspace_id)):
        dist = find_dist(vectors_in_vspace_id[i][1], some_vector)
        navig_dict[dist] = vectors_in_vspace_id[i][0]
        distance_list.append(dist)
    min_dist = min(distance_list)
    print(min_dist)
    min_dist_id = navig_dict[min_dist]
    return vectors_in_vspace_id[min_dist_id]

if __name__ == "__main__":
    #Some tinkering with logistical stuff.
    Orig_to_GreySc()
    Qual_Reduction()
    training_dataset_address = 'Iris_Training_Dataset_GS_CrandRedQ'
    identity_texts = ["Name: Anna Tenikhina\nAge: 20\nLocation: Moscow, Russia\nImage Date: Jan 20, 2016", "Name: Alexandra Yuriyevna\nAge: 42\nLocation: Saint Petersburg, Russia\nImage Date: Jan 17, 2016", "Name: Maria Last_Name_Unknown\nAge: 18\nLocation: Moscow, Russia\nImage Date: Jan 11, 2016", "Name: Battemuulen Naranbat\nAge: 19\nLocation: Amsterdam, Netherlands\nImage Date: Jan 13, 2016", "Name: Anastasiya Last_Name_Unknown\nAge: 21\nLocation: Moscow, Russia\nImage Date: Jan 11, 2016"]
    vectors_in_vspace_id = vectors_in_vspace(training_dataset_address, identity_texts)

    im = Image.open('7A7E9634-E90E-46FD-9AB1-DC22C500750E.JPG')
    gs_im = im.convert('L')
    gs_im.save('Test_GS1.JPG')

    im = Image.open("Test_GS1_Cr.JPG")
    im.save('Test_GS_Cr_Q.JPG', quality=50)

    test_1 = cv2.imread('Test_Dataset/Test_1.JPG')
    test_1_image_array = np.asarray(test_1)
    test_1_RGB_reduced_image = RGB_reduction(test_1_image_array)
    test_1_convolved = convolution_layer(test_1_RGB_reduced_image, features_folder='Features_for_Convolution')
    pooled_list = []
    for item in test_1_convolved:
        pooled_list.append(pooling_layer(item, 13))
    test_1_vector = vectorize_multiple_matrices(pooled_list)

    print(find_match(vectors_in_vspace_id, test_1_vector)[2])

    '''main_list = []
    for item in glob.glob('Iris_Training_Dataset_GS_CrandRedQ/*.JPG'):
        read_image = cv2.imread(item)
        image_array = np.asarray(read_image)
        RGB_reduced_image = RGB_reduction(image_array)
        main_list.append(convolution_layer(RGB_reduced_image, features_folder = 'Features_for_Convolution'))
    print(main_list[0])
    for i in range(5):
        rec_img = image_from_array_reconstruction(main_list[4][i])
        rec_img.save("Test{}.JPG".format(i))
    test_pooled = pooling_layer(main_list[0][0], 13)
    print(test_pooled)
    test_save = np.asanyarray(test_pooled)
    print(test_save)
    test_save = image_from_array_reconstruction(test_save)
    test_save.save('POOLEDTEST.JPG')
    print(vectorize_matrix(test_pooled))
    main_vector = vectorize_matrix(test_pooled)
    some_vector = [0]*len(vectorize_matrix(test_pooled))
    print(some_vector)
    print(len(some_vector) == len(vectorize_matrix(test_pooled)))
    print(find_dist(main_vector, some_vector))
    print(find_dist([3, 0], [0, 4]))

    #TESTING
    test_1 = cv2.imread('Test_Dataset/Test_1.JPG')
    test_1_image_array = np.asarray(test_1)
    test_1_RGB_reduced_image = RGB_reduction(test_1_image_array)
    test_1_convolved = convolution_layer(test_1_RGB_reduced_image, features_folder='Features_for_Convolution')
    pooled_list = []
    for item in test_1_convolved:
        pooled_list.append(pooling_layer(item, 13))
    test_1_vector = vectorize_multiple_matrices(pooled_list)
    print(test_1_vector)

    #some_picture = cv2.imread('Simple_Folder/SS.JPG')
    #pic_to_array = np.asarray(some_picture)
    #print(pic_to_array)
    #output = RGB_reduction("Simple_Folder/SS.JPG")
    #print(output)
    #myarray = np.asarray(output)
    #print(myarray.shape)
    #some_image = image_from_array_reconstruction(output)

    #some_image.save('Test.JPG')
    #for img in glob.glob('{}/*.JPG'.format(folder_name)):'''

    '''blank_list = []
    some_array = pic_to_array[0][0]
    print(some_array)
    for i in pic_to_array:
        pre_processed_img_list = pic_to_array.resize((1, 42, 2),refcheck=False)
        blank_list.append(pre_processed_img_list)
    print(blank_list)'''
    #print("Processed:")
    #print(pre_processed_img_list[0].shape)
    #print(pre_processed_img_list[0])
    #print(cv2.imread('Simple_Folder/SS.JPG'))

    '''img_GS = cv2.imread('Iris_Training_Dataset_GS/Image_GS_1.JPG')
    img_orig = cv2.imread('Iris_Training_Dataset_Orig/IMG_3208.JPG')
    #print(img_orig[0])
    #print(img_GS[0])
    #print(img_GS.shape)
    #sift = cv2.SIFT_create(img)
    #show = plt.imshow(image_data_list[1])
print("Not Processed Yet:")
    print(cv2.imread('Simple_Folder/SS.JPG').shape)
    print(cv2.imread('Simple_Folder/SS.JPG'))
            #output = RGB_reduction("Simple_Folder/SS.JPG")


test1 = image_from_array_reconstruction(output_list_of_matrices[0])
    test2 = image_from_array_reconstruction(output_list_of_matrices[1])
    test3 = image_from_array_reconstruction(output_list_of_matrices[2])
    test4 = image_from_array_reconstruction(output_list_of_matrices[3])
    test5 = image_from_array_reconstruction(output_list_of_matrices[4])
    test1.save('Test1.JPG')
    test2.save('Test2.JPG')
    test3.save('Test3.JPG')
    test4.save('Test4.JPG')
    test5.save('Test5.JPG')

'''