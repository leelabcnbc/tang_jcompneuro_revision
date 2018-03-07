/* this is slower than MATLAB version */
/* so forget about this route */
/*==========================================================
 * arrayProduct.c - example in MATLAB External Interfaces
 *
 * Multiplies an input scalar (multiplier) 
 * times a 1xN matrix (inMatrix)
 * and outputs a 1xN matrix (outMatrix)
 *
 * The calling syntax is:
 *
 *		outMatrix = arrayProduct(multiplier, inMatrix)
 *
 * This is a MEX-file for MATLAB.
 * Copyright 2007-2012 The MathWorks, Inc.
 *
 *========================================================*/
#include <math.h>
#include "mex.h"
/* The computational routine */
void arrayProduct(double *y, double *z, mwSize total)
{
    mwSize i;
    /* multiply each element y by x */
    for (i=0; i<total; i++) {
        z[i] = log(exp(y[i])-1);
    }
}

/* The gateway function */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[])
{
    double *inMatrix;               /* 1xN input matrix */
    size_t ncols;                   /* size of matrix */
    size_t mrows;                   /* size of matrix */
    double *outMatrix;              /* output matrix */

    /* check for proper number of arguments */
    if(nrhs!=1) {
        mexErrMsgIdAndTxt("MyToolbox:inv_softplus:nrhs","One input required.");
    }
    if(nlhs!=1) {
        mexErrMsgIdAndTxt("MyToolbox:inv_softplus:nlhs","One output required.");
    }
    
    /* make sure the second input argument is type double */
    if( !mxIsDouble(prhs[0]) || 
         mxIsComplex(prhs[0])) {
        mexErrMsgIdAndTxt("MyToolbox:arrayProduct:notDouble","Input matrix must be type double.");
    }
    
    /* create a pointer to the real data in the input matrix  */
    inMatrix = mxGetPr(prhs[0]);

    /* get dimensions of the input matrix */
    mrows = mxGetM(prhs[0]);
    ncols = mxGetN(prhs[0]);
    

    /* create the output matrix */
    plhs[0] = mxCreateDoubleMatrix( (mwSize)mrows, (mwSize)ncols, mxREAL);

    /* get a pointer to the real data in the output matrix */
    outMatrix = mxGetPr(plhs[0]);

    /* call the computational routine */
    arrayProduct(inMatrix,outMatrix,(mwSize)ncols*(mwSize)mrows);
}
