import PIL
import PIL
from PIL import Image
import numpy as np
from os import listdir
import glob
import matplotlib.pyplot as plt
import cv2
from scipy.signal import convolve2d

def orig_to_greysc():
    name_value = 1
    # For each file in the Training Dataset folder, opens and greyscales the image. Then, saves it.
    for filename in glob.glob('Iris_Training_Dataset_Orig/*.JPG'):
        im = Image.open(filename)
        gs_im = im.convert('L')
        gs_im.save('Iris_Training_Dataset_GS/Image_GS_{}.JPG'.format(str(name_value)))
        name_value += 1

def qual_reduction():
    # Reduces the quality of a picture by 50% in the cropped Training Dataset folder.
    name_value = 1
    for filename in glob.glob('Iris_Training_Dataset_GS_Cropped/*.JPG'):
        im = Image.open(filename)
        im.save('Iris_Training_Dataset_GS_CrandRedQ/Image_GS_{}.JPG'.format(str(name_value)), quality=50)
        name_value += 1

def RGB_reduction(image):
    ''' Returns a matrix, where each pixel is represented by one number from the greyscale. '''
    # Checks if we are working with arrays.
    if not isinstance(image, np.ndarray):
        # If not, opens the image in an array type.
        opened_img = cv2.imread(image)
        opened_img = np.asanyarray(opened_img)
    else:
        opened_img = image
    # Checks if the image greyscaled.
    if opened_img[0][0][0] == opened_img[0][0][1] and opened_img[0][0][0] == opened_img[0][0][2]:
        # If they are, creates a new matrix (pixel -> number) without the first and second by index channel values,
        # since each pixel has the same three values for each of the RGB channels.
        new_image_matrix = []
        row_numb = 0
        # The construction of the new matrix is made row by row.
        for row in opened_img:
            blank_row = []
            # Pixel by pixel, or number by number.
            for pixel in opened_img[row_numb]:
                new_pixel = pixel[0]
                blank_row.append(new_pixel)
            new_image_matrix.append(blank_row)
            row_numb += 1
    # If not greyscaled, raises a TypeError.
    else:
        raise TypeError("Pixels must be greyscaled first.")
    # Returns the final matrix.
    return new_image_matrix

def image_from_array_reconstruction(matrix):
    # Pass on a matrix, and returns an image.
    blank_matrix = []
    # The construction is done row by row.
    for row in matrix:
        row_count = 0
        blank_row = []
        # Pixel by pixel.
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
    # Convolve() as a helper method that takes two matrices and performs a convolution between them.
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
    # Creates an empty list to hold the images of the features.
    features_list = []
    # Takes each feature, RGB-reduces it, and adds the resulting matrix to the features list.
    # Save sthe feature for future reference as well.
    for item in glob.glob(features_folder + '/*.JPG'):
        feature_img = cv2.imread(item)
        feature_matrix = RGB_reduction(feature_img)
        features_list.append(feature_matrix)
        feature_matrix = image_from_array_reconstruction(feature_matrix)
        feature_matrix.save('feature_matrix1.JPG')
    # Creates an empty list to hold the matrices of the convolution layer.
    output_list_of_matrices = []
    # Performs a convolution between each of the features and the main matrix using the previously defined
    # convolve() method and adds it to the output list.
    for feature_matrix in features_list:
        convolved_matrix = convolve(matrix, feature_matrix)
        output_list_of_matrices.append(convolved_matrix)
    return output_list_of_matrices

def pooling_layer(matrix, dimension):
    # Creates a blank_matrix, which will be the output matrix that is formed from pooling.
    blank_matrix = []
    # The nested loop makes use of the dimension to translate by through the matrix with a period that
    # is equal to it. Hence, we add a dimension to the counters each time we iterate through the loop.
    counter_1 = 0
    for i in range(int(len(matrix)/dimension)):
        blank_row = []
        counter_2 = 0
        # Similar to other methods, it creates the matrix row by row and value by value by taking the mean.
        for j in range(int(len(matrix[0])/dimension)):
            blank_row.append(int(np.mean(matrix[counter_1:(counter_1 + dimension), counter_2:(counter_2 + dimension)])))
            counter_2 += dimension
        blank_matrix.append(blank_row)
        counter_1 += dimension
    return blank_matrix

def vectorize_matrix(matrix):
    # Takes in a matrix and creates a vector by adding its columns one by one.
    blank_vector = []
    matrix = np.asanyarray(matrix)
    for i in range(len(matrix[0])):
        # In doing so, it uses the slicing in numpy arrays and the extend method, which takes in a list and
        # connects it to the existing list (while append adds a list into a list).
        blank_vector.extend(matrix[:, i])
    return blank_vector

def vectorize_multiple_matrices(a_list):
    # By using the vectorize_matrix() helper method, takes in a list of matrices,
    # creates vectors by adding their columns one by one,
    # and then joins those vectors into a one long vector.
    blank_vector = []
    for matrix in a_list:
        micro_blank_vector = vectorize_matrix(matrix)
        blank_vector.extend(micro_blank_vector)
    return blank_vector

def vectors_in_vspace(training_dataset_address, identity_texts):
    # A list to hold all the feature vectors.
    vectors_in_vspace = []
    # For each image in the greyscale training dataset:
    counter = 1
    for file in glob.glob(training_dataset_address + '/Image_GS_*.JPG'):
        print("'{}/Image_GS_{}.JPG' being processed.".format(training_dataset_address, counter))
        # Opens the image.
        image_1 = cv2.imread(training_dataset_address + '/Image_GS_{}.JPG'.format(counter))
        # Creates an array from it.
        image_1_array = np.asarray(image_1)
        # RGB-reduces it into a matrix of values, since the image is already greyscale.
        image_1_RGB_reduced = RGB_reduction(image_1_array)
        # Performs convolution between each of the features and the matrix. Adds to the convolution layer.
        image_1_convolved = convolution_layer(image_1_RGB_reduced, features_folder='Features_for_Convolution')
        # Pools all the matrices in the convolution layer and adds the resulting smaller matrices to a pooled_list.
        pooled_list = []
        for item in image_1_convolved:
            pooled_list.append(pooling_layer(item, 13))
        # Creates a feature vector representing the training image.
        image_vector = vectorize_multiple_matrices(pooled_list)
        # Adds it to the vector space.
        vectors_in_vspace.append(image_vector)
        print("Iteration {} successfully completed.".format(str(counter)))
        counter += 1
    # Creates an empty list for the vectors, but adds in this case a tuple, consisting of an ID number that corresponds
    # to the training image, the feature vector itself, and the identifying text related to the person behind the iris.
    vectors_in_vspace_id = []
    id_numb = 0
    for vector in vectors_in_vspace:
        vectors_in_vspace_id.append((id_numb, vector, identity_texts[id_numb]))
        id_numb += 1
    # Sorts the list of tuples in case it is unordered, which is easily done, since the tuples start integers.
    vectors_in_vspace_id.sort()
    return vectors_in_vspace_id

def find_dist(v1, v2):
    # Helper method that takes two vectors and find the distance between them using np.linalg.norm().
    v1_t_a = np.array(tuple(v1))
    v2_t_a = np.array(tuple(v2))
    return np.linalg.norm(v1_t_a-v2_t_a)

def find_match(vectors_in_vspace_id, some_vector):
    # Creates an empty list to store all the distances from the test feature vector to all the other existing ones.
    distance_list = []
    # In order to be able to access back the information about to whom a certain iris
    # that has the least distance form the test vector belongs, creates a dictionary
    # that has the distances as keys and vector IDs as the values.
    navig_dict = {}
    # For each of the vector in the vector space, accesses the vector's value in the list of tuples vectors_in_vspace_id
    # and finds the distance between that vector and the test one.
    for i in range(len(vectors_in_vspace_id)):
        dist = find_dist(vectors_in_vspace_id[i][1], some_vector)
        # Then, it adds the distance and its corresponding feature vector into the dictionary.
        navig_dict[dist] = vectors_in_vspace_id[i][0]
        # Adds the distance value into the distance_list.
        distance_list.append(dist)
    # Finds the minimum value in the distance_list.
    min_dist = min(distance_list)
    print("\nBest Match at a Distance Score: " + str(min_dist))
    # Accesses the corresponding ID of the image that gives the least distance.
    min_dist_id = navig_dict[min_dist]
    # Returns the tuple of the relevant vector by using min_dist_id as an index,
    # since the vectors are placed into the vector space, having
    # an ID that directly corresponds to their index/order.
    return vectors_in_vspace_id[min_dist_id]

if __name__ == "__main__":
    # Gresyscaling and Quality Reduction of the images in the Training Dataset.
    orig_to_greysc()
    qual_reduction()
    # Clarifies the address of the training dataset.
    training_dataset_address = 'Iris_Training_Dataset_GS_CrandRedQ'
    # Reads and processes the input images' labels from the Training_Dataset_Labels file.
    pre_identity_texts = open("Traning_Dataset_Labels")
    pre_identity_texts = pre_identity_texts.readlines()
    pre_identity_texts = [i for i in pre_identity_texts if i != "\n"]
    counter = 0
    identity_texts = []
    for i in range(int(len(pre_identity_texts) / 4)):
        some_string = "".join(pre_identity_texts[counter: (counter + 4)])
        some_string = some_string[:-1]
        identity_texts.append(some_string)
        counter += 4
    # Creates a vector space based on the feature vectors of the input images.
    vectors_in_vspace_id = vectors_in_vspace(training_dataset_address, identity_texts)
    # Opens the test case from the Test Dataset folder.
    test_1 = cv2.imread('Test_Dataset/Test_1.JPG')
    # Creates an array out of it.
    test_1_image_array = np.asarray(test_1)
    # Reduces the image array into a matrix of values.
    test_1_RGB_reduced_image = RGB_reduction(test_1_image_array)
    # Creates a Convolution Layer using the existing features.
    test_1_convolved = convolution_layer(test_1_RGB_reduced_image, features_folder='Features_for_Convolution')
    #Creates an empty list to hold the outputs matrices of pooling.
    pooled_list = []
    # Executes pooling on each feature map and appends the matrices into the pooled_list.
    for item in test_1_convolved:
        pooled_list.append(pooling_layer(item, 13))
    # Creates a feature vector out of all pooled matrices.
    test_1_vector = vectorize_multiple_matrices(pooled_list)
    # Find the closest match and displays its identificator labels.
    print(find_match(vectors_in_vspace_id, test_1_vector)[2])

    ''' In case it is helpful:
    # Opens the test image and prepares for processing.
    im = Image.open('7A7E9634-E90E-46FD-9AB1-DC22C500750E.JPG')
    gs_im = im.convert('L')
    gs_im.save('Test_GS1.JPG')

    im = Image.open("Test_GS1_Cr.JPG")
    im.save('Test_GS_Cr_Q.JPG', quality=50)
    '''