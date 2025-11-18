import pytest 
import numpy as np
from camera_geometry.image_warper import *
from camera_geometry.projective_space import Homography


# ---------- Nearest Neighbor ---------------------------------------------------

def test_identity_warp_nn():
    image = np.array([[ 69,  91,  83],
                      [ 16, 106,  20],
                      [ 52, 115, 108]])
    identity_map = Homography(np.eye(3))
    identity_warper_nn = ImageWarperNN(identity_map)
    
    warped_image = identity_warper_nn(image)
    assert np.array_equal(warped_image, image)
    
def test_rotation90_warp_nn():    
    image = np.array([[ 69,  91,  83],
                      [ 16, 106,  20],
                      [ 52, 115, 108]])
    theta = np.pi/2 # 90-degree counter-clockwise
    rotation = Homography(np.array([
                    [np.cos(theta), np.sin(theta),   0],
                    [-np.sin(theta),  np.cos(theta),   0],
                    [0, 0,  1.0],
                ]))
    rotation_warper_nn = ImageWarperNN(rotation)
    bg_v = 1 # background_value
    rotated_image = rotation_warper_nn(image,background=bg_v)
    rotated_image_truth = np.array([ image[:,0],         # first column become first row after 90 degrees
                                    [ bg_v, bg_v,  bg_v],
                                    [ bg_v, bg_v, bg_v]])
    assert np.array_equal(rotated_image, rotated_image_truth)
 
 
 # ---------- Bilinear ---------------------------------------------------
 
def test_identity_warp_bi():
    image = np.array([[ 69,  91,  83],
                      [ 16, 106,  20],
                      [ 52, 115, 108]])
    identity_map = Homography(np.eye(3))
    identity_warper_bi = ImageWarperBilinear(identity_map)
    
    warped_image = identity_warper_bi(image)
    assert np.array_equal(warped_image, image)
      
def test_rotation90_warp_bi():    
    image = np.array([[ 69,  91,  83],
                      [ 16, 106,  20],
                      [ 52, 115, 108]])
    theta = np.pi/2 # 90-degree counter-clockwise
    rotation = Homography(np.array([
                    [np.cos(theta), np.sin(theta),   0],
                    [-np.sin(theta),  np.cos(theta),   0],
                    [0, 0,  1.0],
                ]))
    rotation_warper_nn = ImageWarperBilinear(rotation)
    bg_v = 1 # background_value
    rotated_image = rotation_warper_nn(image,background=bg_v)
    rotated_image_truth = np.array([ image[:,0],         # first column become first row after 90 degrees
                                    [ bg_v, bg_v,  bg_v],
                                    [ bg_v, bg_v, bg_v]])
    assert np.array_equal(rotated_image, rotated_image_truth)
    


    
    
    