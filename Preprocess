import cv2
import numpy as np
import os
from math import sqrt,log2
from sklearn.cluster import KMeans

#--------------------constants and stats------------------
pixel_limit=155
folder_name="processed"

min_column=9999
max_column=0
min_row=9999
max_row=0

overall_mean_column_size=0
overall_mean_row_size=0
#-----------------------------------------------


def print_gray(img,filename):
    file = open("texts/"+filename[:-3]+"txt", "w")
    for i in img:
        for j in i:
            file.write(str(j))
            length = len(str(j))
            for x in range(4 - length):
                file.write(" ")
        file.write("\n")
def search(gray,i,j,n,m,neighboor_range):
    low_range_n = max(0, i - neighboor_range)
    low_range_m = max(0, j - neighboor_range)
    high_range_n = min(n - 1, i + neighboor_range)
    high_range_m = min(m - 1, j + neighboor_range)
    counter = 0
    summed=0
    for x in range(low_range_n, high_range_n + 1):
        for y in range(low_range_m, high_range_m + 1):
            if gray[x][y] < pixel_limit:
                counter += 1
                summed+=gray[x][y]
    summed=summed//max(counter,1)
    return counter,(neighboor_range * 2 + 1) ** 2,summed

def weighted_search_vertical(gray,i,j,n,m):
    neighboor_range1 = 4
    neighboor_range2 = 1
    count=0
    summed=0

    for sign_multiplier in [-1,1]:
        increment=sign_multiplier
        # vertical
        first_range_n = min(max(0, i - sign_multiplier*neighboor_range1),n-1)
        second_range_n = min(max(0, i - sign_multiplier*neighboor_range1),n-1)
        first_range_m = max(0, j - neighboor_range2)
        second_range_m = min(m-1, j + neighboor_range2)
        end_flag=False
        for i in range(first_range_n,second_range_n+increment,increment):
            local_count=0
            for j in range(first_range_m,second_range_m):
                if gray[i][j]<pixel_limit:
                    local_count+=1
                    summed+=gray[i][j]
                count+=local_count
                if local_count<2:
                    end_flag=True
                    break
            if end_flag:
                break
        # horizontal
        first_range_m = min(max(0, j - sign_multiplier*neighboor_range1),m-1)
        second_range_m = min(max(0, j - sign_multiplier*neighboor_range1),m-1)
        first_range_n = max(0, i - neighboor_range2)
        second_range_n = min(n-1, i + neighboor_range2)
        end_flag=False
        for j in range(first_range_m,second_range_m+increment,increment):
            local_count=0
            for i in range(first_range_n,second_range_n):
                if gray[i][j]<pixel_limit:
                    local_count+=1
                    summed+=gray[i][j]
                count+=local_count
                if local_count<2:
                    end_flag=True
                    break
            if end_flag:
                break
    if gray[i][j]<pixel_limit:
        summed+=3*gray[i][j]
    summed=summed//max(count,1)
    return count,6+(neighboor_range1*2+1)**2-4*(neighboor_range1-neighboor_range2)**2,summed

def simplify(filename):
    img = cv2.imread('captchas/'+filename)

    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    n,m=gray.shape
    res=np.zeros((n,m,3))
    res=255-res

    case1=case2=base1=base2=base3=base4=line1=0
    for i in range(n):
        for j in range(m):
            #first vetrtical check
            '''counter,area,mean=weighted_search_vertical(gray, i, j, n, m)
            if counter>=area*0.25:
                if gray[i][j]<pixel_limit:
                    mean= gray[i][j]
                res[i][j][0] = res[i][j][1] = res[i][j][2] = 255#mean
                res[i][j][2]=0
                line1+=1
                continue'''

            if gray[i][j]<pixel_limit:
                #mini search

                counter, area, _ = search(gray, i, j, n, m, 1)
                if counter <= 3:
                    res[i][j][0] =res[i][j][1] =res[i][j][2] = 255
                    #res[i][j][0]=0
                    base1+=1
                    continue
                counter, area, _ = search(gray, i, j, n, m, 2)
                if counter <= area * (0.33):  # 7.5->7
                    res[i][j][0] =res[i][j][1] =res[i][j][2] = 255
                    #res[i][j][1] = 0
                    base2+=1
                    continue
                res[i][j][0] = res[i][j][1] = res[i][j][2] = gray[i][j]
                '''res[i][j][0] = res[i][j][1] = res[i][j][2] = gray[i][j]
                counter, area, mean = search(gray, i, j, n, m, 3)
                if counter <= area*0.30:
                    res[i][j][0] =res[i][j][1] =res[i][j][2] = 255
                    res[i][j][2] = 0
                    base3+=1
                    continue

                #main search
                counter,area, _=search(gray,i,j,n,m,4)
                if counter>=area*(0.25):  #%25
                    res[i][j][0] =res[i][j][1] =res[i][j][2]=gray[i][j]
                else:
                    res[i][j][0] =res[i][j][1] =res[i][j][2] = 255
                    #res[i][j][2] = 50
                    res[i][j][0] =0
                    base4+=1'''


            else:
                counter, area, mean = search(gray, i, j, n, m, 1)
                if counter >= 7:
                    res[i][j][0] =res[i][j][1] =res[i][j][2] = mean
                    case1+=1
                    continue
                counter, area, mean = search(gray, i, j, n, m, 2)
                if counter >= area * (0.66):  # 7.5->7
                    res[i][j][0] =res[i][j][1] =res[i][j][2] = mean
                    case2+=1
                else:
                    res[i][j][0] =res[i][j][1] =res[i][j][2]=255

    #print(case1,case2,"-",base1,base2,base3,base4,"-",line1)

    return res

def clip(img):
    #-----------------------clip boundary---------------------
    n,m,_=img.shape
    rows,columns=np.zeros((n)),np.zeros((m))
    column_max_dist=np.zeros((m))
    #----------finding row range--------
    for i in range(n):
        for j in range(m):
            if(img[i][j][0]<pixel_limit):
                rows[i]+=1
    row_counter=column_counter=0
    row_mean=column_mean=0
    epsilon=3
    cut_limit = 0.5

    for i in rows:
        if i>epsilon:
            row_counter+=1
            row_mean+=i
    row_mean=row_mean//row_counter

    row_start = 0
    row_end = n - 1
    for i in range(len(rows)):
        if rows[i] > row_mean * cut_limit and rows[i] > epsilon:
            row_start = i - 2
            break
    for i in reversed(range(len(rows))):
        if rows[i] > row_mean * cut_limit and rows[i] > epsilon:
            row_end = i + 2
            break
    # -----------------------------------

    for j in range(m):
        bottom=-1
        top=9999
        for i in range(row_start,row_end+1):
            if(img[i][j][0]<pixel_limit):
                #pixel count
                columns[j]+=1

                #max dist calculation
                top=i
                if(bottom==-1):
                    bottom=i
        '''if columns[j] >=(row_end-row_start)/2:
            columns[j]*=columns[j]
            #194 1907 1942 1982'''
        if columns[j]>=3:
            columns[j]+=sqrt(top-bottom)

    for i in columns:
        if i>epsilon:
            column_counter+=1
            column_mean+=i
    column_mean=column_mean//column_counter

    column_start = 0
    column_end = m - 1
    for i in range(len(columns)):
        if columns[i] > column_mean * cut_limit and columns[i]>epsilon:
            column_start = i - 5
            break
    for i in reversed(range(len(columns))):
        if columns[i] > column_mean * cut_limit and columns[i]>epsilon:
            column_end = i + 5
            break

    print(filename,"-> means", row_mean, column_mean, "shape", column_end - column_start + 1, row_end - row_start + 1)

    needed_space = 7 - (column_end-column_start+1) % 7
    if (needed_space):
        column_end += needed_space - needed_space // 2
        column_start -= needed_space // 2
    return img[row_start:row_end+1,column_start:column_end+1],columns[column_start:column_end+1]
    #----------------------------------------------------------

def naive_divide(img,filename):
    global min_column,max_column,min_row,max_row,overall_mean_column_size,overall_mean_row_size
    n,m,_=img.shape
    column_size = m // 6
    index,digits=filename.split("-")
    digits=digits.split(".")[0]

    column_start=0

    #-----stat check----
    if(min_column>m):min_column=m
    if(max_column<m):max_column=m
    if(min_row<n):min_row=n
    if(max_row<n):max_row=n
    overall_mean_column_size+=m
    overall_mean_row_size+=n
    #-------------------

    for i in range(6):
        cv2.imwrite(folder_name+"/"+index+","+str(i+1)+"-"+digits[i]+".jpg", img[:,column_start:column_start+column_size])
        column_start += column_size

def k_means(columns,epoch=100):
    m=columns.shape[0]
    #-----center of mass calclation-----
    center_of_mass=np.sum(np.arange(m)*columns)//np.sum(columns)
    #-----------------------------------
    centroids=np.arange(m//7,m,m//7)
    columns=columns                 #increasing column weights to decrease the weight of artifacts if any remaining for k_means

    column_centers=np.zeros(m)
    for i in range(epoch):
        choose_centers(centroids,column_centers)
        new_centroids=center_centroids(columns,column_centers)
        if np.array_equal(centroids,new_centroids):
            print(centroids)
            return centroids
        centroids=new_centroids
    print(centroids)
    return centroids
def choose_centers(centroids,column_centers):
    index=0
    m=column_centers.shape[0]
    while(index<=centroids[0]):
        column_centers[index] = 0
        index+=1
    for high_center in range(1,6):
        while (index < centroids[high_center]):
            if(centroids[high_center]-index<index-centroids[high_center-1]):
                column_centers[index] = high_center
            else:
                column_centers[index] = high_center-1
            index += 1
    while (index < m):
        column_centers[index] = 5
        index += 1

def center_centroids(columns,column_centers):
    m=columns.shape[0]
    centroid_distance_mass_sum=np.zeros(6)
    centroid_mass_sum=np.zeros(6)+0.0000001

    for index in range(columns.shape[0]):
        centroid_distance_mass_sum[round(column_centers[index])] += columns[index]*index
        centroid_mass_sum[round(column_centers[index])]          += columns[index]
    return centroid_distance_mass_sum/centroid_mass_sum






def save_img(img,filename):
    global folder_name
    cv2.imwrite(folder_name+"/" + filename,img)










'''for filename in ["426-613133.jpg"]:
    img = simplify(filename)
    img, columns = clip(img)
    print(img)
    # naive_divide(img,filename)
    save_img(img, filename)
exit()'''

if not os.path.exists(folder_name):
    os.makedirs(folder_name)
for filename in os.listdir("captchas")[1000:1100]:#["1099-010481.jpg"]:#
    img=simplify(filename)
    img,columns=clip(img)
    #naive_divide(img,filename)
    centers=k_means(columns)

    n,m,_=img.shape
    for i in range(n):
        for center in centers:
            center=round(center)
            img[i][center][2]=0
    save_img(img,filename)

print("column-min:",min_column,"max:",max_column,"mean",overall_mean_column_size/100)#len(os.listdir("captchas")))
print("row   -min:",min_row,"max:",max_row,"mean",overall_mean_row_size/100)#len(os.listdir("captchas")))

