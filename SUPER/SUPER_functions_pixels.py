import xarray as xr
import numpy as np
import matplotlib.pyplot as pl


def monthly_correction (ref, input): 
    # Multiplicative bias correction based on monthly climatological values
    # Compute climati
    input_month = input.resample(time='MS').sum().groupby("time.month").mean()
    input_month = input_month.where(input_month != 0, np.nan)

    # input_month
    # Compute ratio per month
    ratio = ref/input_month
    ratio = ratio.where(input_month>=5, 1).where(~np.isnan(ref*input_month))    # if input monthly precip is less than 5mm, then set ratio to 1 5to avoid instability)
    # ratio = ratio.where(np.isnan(ratio), 1)
    # print(np.nanmean(ratio))
    # Apply ratio
    output = input* ratio.sel(month=input.time.dt.month.values).values

    return output


def quadruple_weights(p1,p2,p3,p4):
    # # Compute error (co-)variances of the four precipiation estimates. Only p1 and p2 have cross-correlated errors.
    # (Weights per pixel)


    # Quadruple matrix
    A = np.zeros((13,10))
    A[0, [0, 5]] = 1
    A[1, [1, 6]] = 1
    A[2, [2, 7]] = 1
    A[3, [3, 8]] = 1
    A[4, [4, 9]] = 1
    A[5, 0] = 1
    A[6, 1] = 1
    A[7, 2] = 1
    A[8, 2] = 1
    A[9, 3] = 1
    A[10, 3] = 1
    A[11, 4] = 1
    A[12, 4] = 1

    
    # 
    nt, nlon, nlat = p1.shape
    err_var = np.zeros((nlon,nlat,5))
    for i in range(nlon):
        for j in range(nlat):
            # print(i,j)
            pp1 = p1[:,i,j] - np.nanmean(p1[:,i,j])
            pp2 = p2[:,i,j] - np.nanmean(p2[:,i,j])
            pp3 = p3[:,i,j] - np.nanmean(p3[:,i,j])
            pp4 = p4[:,i,j] - np.nanmean(p4[:,i,j])

            #########################################
            # quadraple
            C11 = np.nanmean(pp1*pp1)
            C22 = np.nanmean(pp2*pp2)
            C33 = np.nanmean(pp3*pp3)
            C44 = np.nanmean(pp4*pp4)
            C12 = np.nanmean(pp1*pp2)
            C13 = np.nanmean(pp1*pp3)
            C14 = np.nanmean(pp1*pp4)
            C23 = np.nanmean(pp2*pp3)
            C24 = np.nanmean(pp2*pp4)
            C34 = np.nanmean(pp3*pp4)

            y = np.array([C11, C22, C33, C44, C12, 
                        C13*C14/C34, C23*C24/C34, C13*C34/C14,
                        C23*C34/C24, C14*C34/C13, C24*C34/C23,
                        C13*C24/C34, C14*C23/C34])
            
            x = np.linalg.inv(A.T @ A) @ A.T @ y            
            

            err_var_p = np.zeros(5)
            err_var_p[0] = x[5] * (x[2] / x[0])
            err_var_p[1] = x[6] * (x[2] / x[1])
            err_var_p[2] = x[7] * (x[2] / x[2])
            err_var_p[3] = x[8] * (x[2] / x[3])
            err_var_p[4] = x[9] * (x[2] / x[4])

            err_var[i,j,:] = err_var_p
    
    return err_var


def merge_data_QC(err_all,products):
    # Merge precipitation data using weighted average based on the quadruple colocation error (co-)variances.
    #
    # 
    data_all = np.array(products)
    

    npdt, nt, nlon, nlat = data_all.shape
    data_merged = np.zeros((nt, nlon, nlat))
    for i in range(nlon):
        for j in range(nlat): 
            # Select a grid point
            err = err_all[i,j,:]
            data = data_all[:,:,i,j]

            # Compute error matrix
            Error = np.array([[err[0], err[4], 0, 0], 
                        [err[4], err[1], 0, 0],
                        [0, 0, err[2], 0],
                        [0, 0, 0, err[3]]])
            
            # Inverse the error matrix
            Error_inv = np.linalg.inv(Error)

            # Compute weights for merging
            lambda_weight = np.sum(Error_inv, axis=1) / np.sum(Error_inv)

            # Check weights validity
            if ((lambda_weight>1) | (lambda_weight<0)).any():
                # If the merging is unstable, ignore the error cross correlation
                # There are better adaptive ways of doing this
                Error = np.diag(err[:4])
                Error_inv = np.linalg.inv(Error)
                lambda_weight = np.sum(Error_inv, axis=1) / np.sum(Error_inv)

            # Weighted average
            data_merged[:,i,j] = np.dot(data.T, lambda_weight).T

    return data_merged



def CTC(p1, p2,p3,x,n):
    # x is the rain/no rain threshold
    # n is the merging parameter



    nt, nlon, nlat = p1.shape
    y1Thres = np.zeros((nt, nlon, nlat))
    for i in range(nlon):
        for j in range(nlat): 
            pp1 = p1[:,i,j]
            pp2 = p2[:,i,j]
            pp3 = p3[:,i,j]

            #flatten the 3D array to 1D array
            data = np.stack((np.ndarray.flatten(pp1),np.ndarray.flatten(pp2),np.ndarray.flatten(pp3)))
            dataThres = -1 + (data > x)*2

            # the inter-product covariance matrix
            covMatrix = np.cov(dataThres)
            # print(i, j)
            Q12=covMatrix.item((0,1))
            Q13=covMatrix.item((0,2))
            Q23=covMatrix.item((1,2))

                
            # quantifing the relative detection skills of each product
            if Q23 != 0 and Q13 != 0 and Q12 != 0:

                v1= np.sqrt((Q12 * Q13)/ Q23)
                v2= np.sqrt((Q12 * Q23)/ Q13)
                v3= np.sqrt((Q13 * Q23)/ Q12)

                # acquiring the weight of each product
                w1 = (v1** n)/(v1**n + v2**n + v3**n)
                w2 = (v2** n)/(v1**n + v2**n + v3**n)
                w3 = (v3** n)/(v1**n + v2**n + v3**n)

                # acquiring a binary rain/norain timeseries with optimal detection skill
                y1 = dataThres.T @ np.array([w1,w2,w3])
                y1Thres[:,i,j] = -1 + (y1 > x)*2

    return y1Thres.reshape(p1.shape)

