K Means

    1.Generate 3 clusters of different size and shape

        X_1 = np.random.multivariate_normal(mean=[4, 0], cov=[[1, 0], [0, 1]], size=75)

        X_2 = np.random.multivariate_normal(mean=[6, 6], cov=[[2, 0], [0, 2]], size=250)

        X_3 = np.random.multivariate_normal(mean=[1, 5], cov=[[1, 0], [0, 2]], size=20)

    2.Observe the result of k-mean clustering for different initial positions of centres.

    3.Run GMM on same data and compare both methods

    4.In this part, we’ll implement k-means to compress an image.<br> Take a high resolution RGB image. Therefore, for each pixel location we would have 3 8-bit integers that specify the red, green, and blue intensity values. Our goal is to reduce the number of colors to 30 and represent (compress) the photo using those 30 colors only. To pick which colors to use, we’ll use k-means algorithm on the image and treat every pixel as a data point in 3-dimensional space which is the intensity of RGB. Will run kmeans to find 30 centroids of colors and finally we will represent the image using the 30 centroids for each pixel. Check if the actual size of the image changes.
