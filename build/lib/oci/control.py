#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 19 09:15:39 2026
@author: Lara Colognese de Almeida
"""
#%% Header: import libraries

from scipy import signal  # signal processing library
import numpy as np  # important package for scientific computing
import pysid as ps  # system identification package
import vrft # vrft package
import oci  # oci package

#%% Function that design the controller with the OCI method

def design(u, y, Td, Cf, model, model_params, L):
    # Description of the design function to help the user
    """Function that design the controller using the OCI method.
    Inputs: u,y,Td,Cf,model,model_params,L
    Output: C
    
    Inputs description:
        u: input data matrix. The dimension of u must be (N,n), where N is the data length and n is the number of inputs/outputs of the system;
        
        y: output data matrix. The dimension of y must be (N,n), where N is the data length and n is the number of inputs/outputs of the system;
        
        Td: Reference Model transfer matrix. It must be a python list of TransferFunctionDiscrete elements. The dimension of the list must be (n,n);
        
        Cf: Fixed part of the controller structure that will be used on the method.  It must be a python list of TransferFunctionDiscrete elements. The dimension of the list must be (n,n);
        
        model: identification model used to estimate C_til, the inverse of the non-fixed part of the controller (C_til = 1/C_i). The options are: 

          model     Description                                            Formulation
          _____________________________________________________________________________________________
          'OE'     (Output-Error)                                         y(t) = [B(z)/F(z)]u(t) + e(t), 
          'ARX'    (Auto-Regressive with eXogenous input)                 A(z)y(t) = B(z)u(t) + e(t), 
          'ARMAX'  (Auto-Regressive Moving Average with eXogenous input)  A(z)y(t) = B(z)u(t) + C(z)e(t), 
          'BJ'     (Box-Jenkins)                                          y(t) = [B(z)/F(z)]u(t) + [C(z)/D(z)]e(t), 
        
        model_params: parameters of the identification model.

            It must always be a matrix (list of lists) with dimension (ny, nu), where each
            element [i][j] is a list of parameters corresponding to the model relating input j 
            to output i.

            The format of each parameter list depends on the selected model:

                - 'OE':    [nb, nf, nk]
                - 'ARX':   [na, nb, nk]
                - 'ARMAX': [na, nb, nc, nk]
                - 'BJ':    [nb, nc, nd, nf, nk]

            Where:
                na: polynomial order of A(z) (output dynamics)
                nb: polynomial order of B(z) (input dynamics)
                nc: polynomial order of C(z) (noise model)
                nd: polynomial order of D(z) (noise model denominator)
                nf: polynomial order of F(z) (input dynamics denominator)
                nk: input-output delay (number of samples)

            Example (ny=2, nu=2):
                model_params = [
                    [ [nb_11, nf_11, nk_11], [nb_12, nf_12, nk_12] ],
                    [ [nb_21, nf_21, nk_21], [nb_22, nf_22, nk_22] ]
                ]

            For SISO systems (ny = 1, nu = 1), the structure must still be respected:
                model_params = [ [ [nb, nf, nk] ] ]

        L: OCI method filter. It also must be a python list of TransferFunctionDiscrete elements. The dimension of the list must be (n,n).
        
    Outputs description:
        C: controller obtained by the OCI method. C = C_f * C_i, where:
          C_f: fixed part of the controller structure that will be used on the method. It is the same as the input Cf.
          C_i: non-fixed part of the controller structure that will be estimated by the method.

    """

    # Validate dimensions and model inputs
    nu = u.shape[1] if u.ndim > 1 else 1
    ny = y.shape[1] if y.ndim > 1 else 1
    oci.validate_model_inputs(model, model_params, ny, nu)

    # Tests for the SISO scenario:
    # testing the type of Td set by the user and converting it to list
    if isinstance(Td, signal.dlti):
        Td = [[Td]]
    # testing the type of Cf set by the user and converting it to list
    if isinstance(Cf, signal.dlti):
        Cf = [[Cf]]
    # testing the type of L set by the user and converting it to list
    if isinstance(L, signal.dlti):
        L = [[L]]

    # number of data samples/ data length
    N = len(u)
    # number of inputs/outputs of the system -> caso multi variável
    n = len(Td)
    # creates a dummy time vector, necessary for the vrft.stbinv function
    t = np.linspace(0, N - 1, N)  # linspace(start,stop,numberofpoints)
    # pushing the vector to have the specified dimensions
    t.shape = (1, N)

    # Filter the signal u
    uf = vrft.filter(L, u)

    # Calculates u_td = Td*u_f
    u_td = vrft.filter(Td, uf)

    # transformation of Cf from the MIMO transfer function list structure to a state-space model
    Acf, Bcf, Ccf, Dcf = vrft.mtf2ss(Cf)
    
    # Calculates pre_u = (Cf)^-1 * u_td
    pre_u, _, flag_1 = vrft.stbinv(Acf, Bcf, Ccf, Dcf, u_td.T, t)
    pre_u = pre_u.T

    if flag_1 == 1:
        # if flag=1, then it was not possible to calculate the inverse of the fixed control model. OCI method aborted!
        print(
            "It was not possible to calculate the virtual fixed control. The inversion algorithm has failed."
        )
        # throw an error 
        raise RuntimeError(
            "It was not possible to calculate the virtual fixed control. The inversion algorithm has failed."
        )

    elif flag_1 == 2:
        # if flag=2, the inverse of the fixed control model is unstable. OCI method aborted!
        print(
            "The inverse of the fixed control model Cf(z) is unstable. It is not recommended to proceed with the OCI method. The algorithm was aborted!"
        )
        raise RuntimeError(
            "The inverse of the fixed control model Cf(z) is unstable. It is not recommended to proceed with the OCI method. The algorithm was aborted!"
        )
    
    # Ident matrix
    I = oci.eye_tf(n, dt=1)

    # Calculates one_minus_Td = (1-Td)
    one_minus_Td = oci.subtract_tf(I, Td) 

    # transformation of (1 - Td) from the MIMO transfer function list structure to a state-space model
    A_one_minus_Td, B_one_minus_Td, C_one_minus_Td, D_one_minus_Td = vrft.mtf2ss(one_minus_Td)
    
    # calculates u_til = (1 - Td)^-1 * pre_u
    u_til, _, flag_2 = vrft.stbinv(A_one_minus_Td, B_one_minus_Td, C_one_minus_Td, D_one_minus_Td, pre_u.T, t)
    u_til = u_til.T

    if flag_2 == 1:
      # if flag=1, then it was not possible to calculate the inverse of (1-Td). OCI method aborted!
      print(
          "It was not possible to calculate the inverse of (1 - Td). The inversion algorithm has failed."
      )
      # throw an error 
      raise RuntimeError(
          "It was not possible to calculate the inverse of (1 - Td). The inversion algorithm has failed."
      )

    elif flag_2 == 2:
        # if flag=2, the inverse of (1-Td) model is unstable. OCI method aborted!
        print(
            "The inverse of (1-Td) is unstable. It is not recommended to proceed with the OCI method. The algorithm was aborted!"
        )
        raise RuntimeError(
            "The inverse of (1-Td) is unstable. It is not recommended to proceed with the OCI method. The algorithm was aborted!"
        )

    # test if the inversion algorithms were succesful
    if flag_1 == 0 and flag_2 == 0:
        # if flag_1=0 and flag_2=0, then, the inversion algorithms were succesful
        # remove the last samples of y, to match the dimensions of the u_til
        # number of samples used in the method
        
        N = u_til.shape[0]
        y = y[0:N, :]

        # obtain the size
        ny = y.shape[1]
        nu = u_til.shape[1]

        if model == 'OE':
            # recover the parameters
            params = oci.unpack_model_params('OE', model_params)            
            model = ps.oe( params['nb'],  params['nf'], params['nk'], u_til, y) 
            # identify C_til = 1/C_i           
            C_til = [[signal.TransferFunction(
                  oci.clean_coefficients(model.B[i][j]), 
                  oci.clean_coefficients(model.F[i][j]), dt=1)
              for j in range(nu)] for i in range(ny)]

        elif model == 'ARX':
            # recover the parameters
            params = oci.unpack_model_params('ARX', model_params)
            model = ps.arx(params['na'], params['nb'], params['nk'], u_til, y)
            # identify C_til = 1/C_i 
            C_til = [[signal.TransferFunction(
                  oci.clean_coefficients(model.B[i][j]),
                  oci.clean_coefficients(model.A[i][j]), dt=1)
              for j in range(nu)] for i in range(ny)]

        elif model == 'ARMAX':
            # recover the parameters
            params = oci.unpack_model_params('ARMAX', model_params)
            model = ps.armax(params['na'], params['nb'], params['nc'], params['nk'], u_til, y)
            # identify C_til = 1/C_i 
            C_til = [[signal.TransferFunction(
                  oci.clean_coefficients(model.B[i][j]),
                  oci.clean_coefficients(model.A[i][j]), dt=1)
              for j in range(nu)] for i in range(ny)]

        elif model == 'BJ':
            # recover the parameters
            params = oci.unpack_model_params('BJ', model_params)
            model = ps.bj(params['nb'], params['nc'], params['nd'], params['nf'], params['nk'], u_til, y)
            # identify C_til = 1/C_i
            C_til = [[signal.TransferFunction(
                  oci.clean_coefficients(model.B[i][j]),
                  oci.clean_coefficients(model.F[i][j]), dt=1)
              for j in range(nu)] for i in range(ny)]
      
        # Calculates the inverse of C_til
        inv_C_til, flag_3 = oci.invert_mtf(C_til)
        
        # test if the inversion algorithm was succesfull
        if flag_3 == 1:
          # if flag=1, then it was not possible to calculate the inverse of (C_til). OCI method aborted!
          print(
           "It was not possible to calculate the inverse of (C_til). The inversion algorithm has failed."
          )
          raise RuntimeError(
          "It was not possible to calculate the inverse of (C_til). The inversion algorithm has failed."
          )
        
        if flag_3 == 2:
          # if flag=2, the inverse of (C_til) model is unstable. Just a warning!
          print(
           "The inverse of (C_til) is unstable. It is not recommended to proceed with the OCI method. The algorithm was aborted!"
          )

        if flag_3 == 0 or flag_3 == 2:
          # calculates Cf*C_i
          C = oci.multiply_tf(Cf, inv_C_til)
          # return the controller calculated with the OCI method
          return C