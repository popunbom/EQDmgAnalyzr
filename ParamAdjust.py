# -*- coding: utf-8 -*-
# Author: Fumiya ENDOU <fantom0779@gmail.com>
# Created by PyCharm at 2019-03-10

# This is a part of EQDmgAnalyzr

import math
from abc import ABCMeta, abstractmethod
from time import sleep
import matplotlib
import matplotlib.pyplot as plt
import cv2
import numpy as np


class ParamAdjust(metaclass=ABCMeta):


  class Trackbar():
    
    def get_label( self ):
      return self.label
    
    def get_val( self ):
      return cv2.getTrackbarPos( self.label, self.wnd_name )
    
    # def is_changed( self ):
    #   retVal = self.changed
    #   self.changed = False
    #   return retVal
    
    # def update( self, _ ):
    #   self.val = cv2.getTrackbarPos( self.label, self.wnd_name )
    #   self.changed = True
    
    def __init__( self, wnd_name, update_callback, *, label, val_range ):
      assert type( wnd_name ) == str, f"'wnd_name' must be str (type(wnd_name) = {type( wnd_name )}"
      assert type( label ) == str, f"'label' must be str (type(label) = {type( label )}"
      assert type( val_range ) == tuple, f"'val_range' must be tuple (type(val_range) = {type( val_range )}"
      
      
      self.label = label
      self.range = val_range
      self.wnd_name = wnd_name
      
      # self.val = 0
      # self.changed = False
      
      cv2.createTrackbar( label, wnd_name, *val_range, update_callback )
  
  
  class SubImage():
    
    def get_label( self ):
      return self.label
    
    def get_img( self ):
      return self.img
    
    def get_imshow( self ):
      return self.imshow
    
    
    def __setattr__(self, key, value):
      if key == 'img' and value is not None:
        print("'img' has changed")
        if value.ndim == 2:
          self.subplt.imshow( value, cmap=self.cmap )
        elif value.ndim == 3:
          self.subplt.imshow( cv2.cvtColor( value, cv2.COLOR_BGR2RGB ), cmap=self.cmap )
        self.subplt.figure.canvas.draw()
        self.subplt.figure.canvas.flush_events()
      super().__setattr__(key, value)
    
    def __init__( self, label, img, subplt, *, cmap='gray' ):
      assert type( label ) == str, f"'label' must be str (type(label) = {type( label )}"
      assert img is None or type( img ) == np.ndarray, f"'img' must be None or numpy.ndarray (type(img) = {type( img )}"
      assert type( cmap ) == str, f"'cmap' must be str (type(cmap) = {type( cmap )}"
      # assert type( subplt ) == matplotlib.axes._subplots.AxesSubplot, f"'subplt' must be matplotlib.axes._subplots.AxesSubplot (type(subplt) = {type(subplt )}"
      
      
      self.label = label
      self.cmap = cmap
      self.subplt = subplt
      self.img = img
      
      self.subplt.set_xticks( [] )
      self.subplt.set_yticks( [] )
      self.subplt.set_title( self.label )


  def get_trackbar( self, *names ):
    assert len( names ) != 0, "There must be at least 1 argument."
    return tuple( [self.trackbars[name] for name in names] )
  
  
  def get_subimage( self, *names ):
    assert len( names ) != 0, "There must be at least 1 argument."
    return tuple( [self.images[name] for name in names] )
  
  
  
  def __init__( self, trackbars, images, *, name_images_window="Figures", name_trackbar_window="Trackbars",
                tuple_shape=None ):
    assert type( trackbars ) == dict, f"'trackbars' must be dict (type(trackbars) = {type( trackbars )}"
    assert type( images ) == dict, f"'images' must be dict (type(images) = {type( images )}"
    assert type( name_images_window ) == str, f"'name_images_window' must be str (type(name_images_window) = {type(name_images_window )}"
    assert type( name_trackbar_window ) == str, f"'name_trackbar_window' must be str (type(name_trackbar_window) = {type(name_trackbar_window )}"
    assert tuple_shape is None or type( tuple_shape ) == tuple, f"'tuple_shape' must be None or str (type(tuple_shape) = {type( tuple_shape )}"
    
    # plt.ion()
    plt.ioff()
    plt.suptitle( name_images_window )
    cv2.namedWindow(name_trackbar_window)
  
    if tuple_shape is None:
      nrows = math.ceil( math.sqrt( len( images ) ) )
      ncols = math.ceil( len( images ) / nrows )
    else:
      ncols, nrows = tuple_shape
    
    for index, (k, v) in enumerate( images.items(), start=1 ):
      images[k]['subplt'] = plt.subplot( ncols, nrows, index )
    
    self.trackbars = { k: ParamAdjust.Trackbar( name_trackbar_window, self.update, **v ) for (k, v) in trackbars.items() }
    self.images = { k: ParamAdjust.SubImage( **v ) for (k, v) in images.items() }
    
    self.update()
    
    
  def run(self):
    plt.show()
  
  @abstractmethod
  def update( self, *args ):
    raise NotImplementedError
