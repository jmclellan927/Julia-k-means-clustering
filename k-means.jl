#= Chooses 'n' random points from an array of data points where each row is a
different data point and outputs them.
 Inputs:
 features: The raw data of each data point without labels sorted such that each
 row is a new data point, each column is a value corresponding to that data
 point.

 n: The number of random points the user would like to generate.

 Outputs:
 centr: An array (n by length(features[1,:]) dimensions) of random points from
 the data set inputted.

 Example:
 Let 'features' be the data from the iris dataset, with 3 random points to be
 selected:
    IntCentrs(features,3)

 Outputs:
    3×4 Array{Float64,2}:
    6.1  2.9  4.7  1.4
    5.5  2.4  3.8  1.1
    6.3  2.8  5.1  1.5
 =#
function IntCentrs(features,n)
    centr = zeros(n,length(features[1,:]))
    for i = 1:n
        centr[i,:] = features[rand(1:length(features[:,1])),:]
    end
    return centr
end
#______________________________________________________________________________#

#______________________________________________________________________________#

 #=The cluster function finds the centroids of 'n' user defined clusters. This
 function initializes the centroids, then calculates the distance between the
 centroids and each data point. The minimum distance is used to assign a data
 point to a centroid. The corresponding data points are averaged, thus
 calculating a new centroid. The function iterates through this method until the
 centroids stop moving, indicating that the centroid for each cluster is
 identified. It then outputs the centroids, the labels of each centroid, and
 the label of the query.

 Inputs:
 features: The raw data of each data point sorted such that each row is a new
 data point, each column is a value corresponding to that data point.

 n: The number of clusters the user would like to find.

 query: The unlabeled data point that you would like to identify.

 labels: The labels of the data points in features.

 Outputs:
 centroidID: The calculated centroids, an array of (n by length(features[:,1]))
 dimensions.

 centroidLabels: The labels of each centroid.

 queryLabel: The label that was assigned to the inputted query.

 Example:
 Let 'features' be the raw data from the iris dataset, and the user wants to find
 3 clusters, and identify the point [6,4,2,4].

 cluster(features,3,[6,4,2,4],labels)

 Output:
 ([5.006 3.428 1.462 0.246;
 5.90161 2.74839 4.39355 1.43387;
 6.85 3.07368 5.74211 2.07105],

 ["setosa"; "versicolor"; "virginica"],

 "versicolor")
 =#
function cluster(features,n,query,labels)
    #=I begin by calling the IntCentrs function, which generates 'n' centroids
    by pulling random points from the data set. Then, I create an empty array
    to store the new centroids. These are also used to enter the while-loop that
    will iterate until the centroids converge on a solution. For the
    newCentroids array, each row corresponds to a new centroid, and each column
    is a different dimension of centroid.=#
    centroidID = IntCentrs(features,n)
    newCentroids = zeros(n,length(features[1,:]))

    #=centroidLabels stores the labels for each centroid once they are calculated.
    Each row in centroidLabels corresponds to each row in centroidID such that
    the row index pairs the centroid to the correct label. =#
    centroidLabels = fill("",length(centroidID[:,1]),1)

    #=queryDist will store the distances between the query and all the centroids
    once the centroids are calculated so the minimum distance, and thus the
    cluster to which the query belongs, can be found.=#
    queryDist = zeros(1,length(centroidID[:,1]))

    #=queryLabel contains the label of the query, thus identifying it.=#
    queryLabel = ""

    #=This while-loop iterates until the centroids stop moving.=#
    while true
        #=Since the newCentroids array is updated with each iteration, I empty
        it before beginning the next iteration so the values from the previous
        run do not affect the calculation.=#
        newCentroids = zeros(n,length(features[1,:]))

        #=I now create an empty array that counts how many data points are
        associated with each centroid, to be used for calculating the new
        centroids. Each column corresponds to a new centroid.=#
        centroidCount = zeros(1,n)

        #=Now I create an empty array that will store the distances of each data
        point to each centroid.=#
        distances = zeros(length(features[:,1]),length(centroidID[:,1]))

        #=closePt is an array which stores information about closest point,
        (min distance, centroid ID)=#
        closePt = zeros(1,1)

        #=This for-loop iterates through each data point in 'features' and
        calculates the distance between each centroid.=#
        for i = 1:length(features[:,1])

            #=This for-loop iterates through each centroid, calculates the
            distance between the data point at row 'i' of the features and
            centroid 'j'. The distances are stored such that each row corresponds
            to a data point, and each column contains the distance to a centroid.=#
            for j = 1:length(centroidID[:,1])
                distances[i,j] = dist(features[i,:],centroidID[j,:])
            end

            #=closePt stores the minimum distance and location of the minimum
            distance in a tuple. Since each column represents a different
            centroid the returned location of the minimum distance is the ID
            of the centroid it is associated with.=#
            closePt = findmin(distances[i,:])

            #=The 'newCentroids' array is updated such that the features of the
            data point that corresponds to the minimum distance are added to
            the row corresponding to the closest data point.=#
            newCentroids[closePt[2],:] = newCentroids[closePt[2],:] + features[i,:]

            #=centroidCount counts the number of data points that associated
            with each centroid so that the mean may be calculated later. Each
            time a new feature is added above, the column corresponding to that
            centroid is increased by 1.=#
            centroidCount[1,closePt[2]] = centroidCount[1,closePt[2]] + 1
        end

        #=Each column in 'newCentroids' contains the summed values of each
        dimension of the points associated with each centroid, where each row
        corresponds to each centroid. This for-loop iterates through each row,
        and divides each column by the number of data points that
        were added to that centroid ID, so that the mean of each cluster is
        stored in the newCentroids array, thus giving the final value of the
        centroids for this iteration.=#
        for i = 1:n

            #=This if construct makes sure that there were any data points
            associated with the centroid at all. If not, it does not update that
            centroid do avoid dividing by 0.=#
            if centroidCount[i] != 0
                newCentroids[i,:] = newCentroids[i,:]./centroidCount[i]
            end
        end

        #=This if-construct checks if there is a difference between the new
        centroids and the old centroids. I subtract each element in the
        newCentroids array from the old centroids in centroidID. I then sum all
        of these elements and take the absolute value of the sum of this value.
        If the two arrays are identical, I get an array full of 0s so the sum
        will be 0. If this happens, I call knn to identify
        the label of each centroid and the while-loop stops.=#
        if abs(sum(newCentroids - centroidID)) == 0.0
            for i = 1:length(centroidID[:,1])
                #=call knn and put the returned label in the index of
                centroidLabels that corresponds to the centroid at that same
                index in centroidID=#
                centroidLabels[i] = knn(centroidID[i,:],features,labels,5)[2]
            end

            #=Iterate through centroidID and calculate the distance between
            the query and each centroid.=#
            for i = 1:length(centroidID[:,1])

                #=Stores the distances from the query to each centroid.=#
                queryDist[i] = dist(query,centroidID[i,:])
            end

            #=queryLabel is set as the label of the centroid to which the query
            is closest.=#
            queryLabel = centroidLabels[argmin(queryDist)[2]]
    #=Break from the while loop=#
    break
        end

        #=The values stored in 'newCentroids' are copied to 'centroidID' so that
        the next iteration may clear the 'newCentroids', but still use these
        centroids in calculating the new centroids.=#
        centroidID = copy(newCentroids)

    end

    return centroidID, centroidLabels, queryLabel
end

testCenter = cluster(features,3,[6,4,2,4],labels)
#______________________________________________________________________________#

#______________________________________________________________________________#
 #=This function calculates the sum of squared errors and plots them in order to
 find the elbow to determine the correct number of clusters. The user simply
 inputs the dataset they would like to use and the maximum number of clusters they
 would like to find the sum of squared errors for. The function iterates from 1
 up to this number.
 Inputs:
 features: The raw data of each data point sorted such that each row is a new
 data point, each column is a value corresponding to that data point.

 numClusters: The maximum number of clusters the user would like to find the
 sum of squared errors for.

 Outputs:
 SSE: A vector of the sum of squared errors for clusters from 1 up to the
 input for numClusters.

 Example:
 Let 'features' be the raw data from the iris dataset without labels, and 10
 be the maximum number of clusters the user would like to examine.

 elbowMethod(features,10)

 Output:
 1×10 Array{Float64,2}:
 681.371  152.348  78.8514  57.256  …  36.1668  36.8391  31.3267

  -- A plot of the output (SSE) vs. the number of clusters.
 =#
function elbowMethod(features,numClusters)

    #=SSE is an zeros vector that will store the SSE for each value 'k'=#
    SSE = zeros(1,numClusters)
    labels = fill("",1,length(features))
    #=The first for-loop iterates through the maximum number for clusters the
    user inputs. If they would like to iterate through 10 clusters, this
    loop through all clusters up to 'n'.=#
    for n = 1:numClusters
        #=For each iteration, the 'cluster' function is called to find the
        centroids for 'n' clusters.=#
        centroidID = cluster(features,n,[],labels)[1]
        #println(centroidID)
        #=This for-loop iterates through each data point in the 'features'
        dataset, finds the closest centroid, calculates the distance between
        the data point and that centroid, and adds that value to the SSE array
        to be plotted at the end of the function.=#
        for i = 1:length(features[:,1])

            #=This is a zeros vector that will store the distances between each
            centroid.=#
            distances = zeros(1,length(centroidID[:,1]))

            #=This for loop iterates through each centroid and calculates the
            distance between centroid 'j' and data point 'i'. The distances are
            stored in the distances array.=#
            for j = 1:length(centroidID[:,1])
                distances[j] = dist(centroidID[j,:],features[i,:])
            end

            #=The SSE array is updated at the position 'n', for the number
            of clusters in this iteration. The minimum distance means the point
            belongs to that cluster, so that value is the one used to compute
            the SSE. =#
            SSE[n] = SSE[n] + findmin(distances)[1]
        end
    end

    #A plot is generated for each trial of 'k' clusters and their SSE.
    PlOt = plot(1:numClusters,SSE[:],title = "Elbow Method",xlabel = "Clusters", ylabel = "SSE", label = "")
    savefig("ElbowPlot.png")

    return SSE
end

elbowMethod(features,10)
#=
=#
#______________________________________________________________________________#

#______________________________________________________________________________#

 #=Calculate the silhouette coefficient for a 'features', a raw dataset without
 labels, and 'n' clusters. Function then iterates through each point 'i' in each
 cluster and calculates the distance from that point and each point in its same
 cluster. It then calculates the distance to the points in the next closest
 cluster and finds the average. It then calculates the silhouette
 coefficient 'si' for this data point from this minimum distance. After finding
 si for point each point 'i' calculates the average Si for all data points.
 Input:
 features: The raw data of each data point sorted such that each row is a new
 data point, each column is a value corresponding to that data point.

 n: The number of clusters the user would like to examine.

 Output:
 Si: Average silhouette coefficient for all points.

 Example:
 Let 'features' be the raw data from the iris dataset without labels, and 5
 be the number of clusters the user would like to examine.

 silhouette(features,5)

 Output:
 0.5614297356366115
 =#
function silhouette(features,n)

    labels = fill("",1,length(features))
    #=Create the centroids and store them in centroidID. Each row in this array
    corresponds to a different centroid.=#
    centroidID = cluster(features,n,[],labels)[1]

    #=featureID is a 1D column vector that will store the centroid associated with
    each point. i.e. it will store '1' if the centroid associated with point i
    is the first row of centroidID.=#
    featureID = zeros(Int,length(features[:,1]),1)

    #=nextClosestID is a 1D column vector that will store the next closest
    centroid to data point i. This is to be used in calculating b(i)=#
    #nextClosestID = zeros(Int,length(features[:,1]),1)

    #=nextClosestDistances =#
    nextClosestDistances = zeros(1,length(centroidID[:,1]))

    #=Initialize a and b. a stores the sum of the minimum distances of point i
    to each other point in its same cluster. b stores the distance of point i
    to each other point that is not in its same cluster.=#
    a = 0.0
    b = 0.0

    #=Initialize Silhouette coefficient. si is the silhouette coefficient for
    each data point. Si will be returned at the end of the function and is the
    average of all the silhouette coefficients, si, for each data point.=#
    Si = 0.0
    si = 0.0

    #=centroidCount counts the number of points associated with each centroid in
    centroidID. It is a 1D row vector where each element corresponds to the
    centroid held in that row of centroidID.=#
    centroidCount = zeros(1,length(centroidID[:,1]))

    #=This for-loop iterates through each data point associates each data point
    i with the closest centroid in centroidID. It stores the associated ID in
    featuresID, as explained above.=#
    for i = 1:length(features[:,1])

        #=The distances vector stores the distances between the data point and
        each centroid.=#
        distances = zeros(1,length(centroidID[:,1]))

        #=This for-loop calculates the distances from each centroid to each
        data point and stores them in the distances vector.=#
        for j = 1:length(centroidID[:,1])
            distances[j] = dist(centroidID[j,:],features[i,:])
        end

        #=Store the centroid associated with feature 'i' in featureID. The
        'trunc' method simply is used to truncate and convert the
        cartesian coordinate into an integer so it may be stored in featureID.
        argmin identifies the element index that has the smallest distance to
        feature i. The index of this distance corresponds to the centroid at
        that index.=#
        featureID[i] = trunc(Int,argmin(distances)[2])

        #=I now update the centroidCount to keep track of how many points are
        in each cluster.=#
        centroidCount[argmin(distances)] = centroidCount[argmin(distances)] + 1

    end

    #=This for-loop calculate a(i) and b(i) by iterating through all the data
    points. It iterates through each data point 'i', and compares it to each
    other data point 'j' in the data set. If they belong to the same cluster, it
    calculates the distance and adds it to a. if they are not in the same
    cluster, it calculates the distance and adds it to b. At the end, it
    calculates the average of these two and calculates the silhouette
    coefficient.=#
    for i = 1:length(features[:,1])
        #=I reset a and b at the beginning of each loop so I can calculate the
        silhouette coefficient for the next element.=#
        a = 0.0
        b = 0.0


        nextClosestDistances = zeros(1,length(centroidID[:,1]))


        #=This for-loop iterates through each data point 'j' to compare them to
        data point 'i'.=#
        for j = 1:length(features[:,1])

            #=If 'j' and 'i' are the same data point, I do nothing=#
            if j == i

            #=If the featureID for data point i matches data point j, that is,
            they belong to the same cluster, I calculate the distance and add it
            to a=#
            elseif featureID[i] == featureID[j]
                a = a + sqrt(dist(features[i,:],features[j,:]))

            #=If featureID for points i and j do not match, then I store the
            distance from point i to j in the index associated with the cluster
            that j is a part of=#
            elseif featureID[i] !=featureID[j]

                #=I add the distance from i to j to nextClosestDistances at
                the cluster index of j. The summation will allow me to easily
                calculate the average distance to all points in each cluster
                after each point is iterated through.=#
                nextClosestDistances[featureID[j]] = nextClosestDistances[featureID[j]] + sqrt(dist(features[i,:],features[j,:]))
            end
        end

        #=By default, the minimum distance in nextClosestDistance will be 0, the
        which is at the index belonging to data point i. This value is set to
        an astronomically high value so the second closest distance can be found.
        =#
        nextClosestDistances[featureID[i]] = 1e10

        #=The sum of all distances in nextClosestDistances are divided by the
        number of data points in the cluster to which they belong to calculate
        the average distance between data point i and the points in each cluster.
        =#
        nextClosestDistances = nextClosestDistances./centroidCount
        #println(centroidCount)

        #TESTING THIS EDIT! Setting NaN to inf so 'b' does not become NaN.
        for j = 1:length(nextClosestDistances)
            if isnan(nextClosestDistances[j])
                 nextClosestDistances[j] = Inf
            end
        end

        #=b is set to the minimum average distance in nextClosestDistance.=#
        b = findmin(nextClosestDistances)[1]
        #println(nextClosestDistances)
        #println(b)

        #=Now I divide a by the number elements in its cluster to calculate the
        average to be used in calculating the silhouette coefficient. I
        subtract the centroidCount for this feature by '1' because one of the
        data points in centroidCount is the data point whose 'a' I am
        calculating.=#
        if centroidCount[featureID[i]] > 1
            a = a/(centroidCount[featureID[i]]-1)
        end
        #println(a)
        #println(max(a,b))
        #=Calculate the coefficient number for data point i=#
        si = (b-a)/max(a,b)
        #println(b)
        #println(max(a,b))
        #println(si)
        #=Stores the sum of all the silhouette coefficient for each data point.
        This is so the average of all silhouette coefficients may be calculated
        later.=#
        Si = Si + si
        #println(si)
        #println(Si)
    end

    #=Averages the silhouette coefficient. This is the value that is returned
    after the function iterates through each data point.=#
    Si = Si/length(features[:,1])

    return Si
end
