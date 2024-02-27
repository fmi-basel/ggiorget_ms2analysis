# @ String link_dist
# @ String gap_closing_dist
# @ String ngap_max

# This script is to run TrackMate with StarDist cell segmentation on an open image to
# generate cell masks that are linked in time. The tracking parameters can be put in
# through an interface. The output files (Trackmate xml instance and label image) 
# will be saved in the same directory as the input image. Works well, if cells are
# 15-30px in diameter. Units for input values are in units of the image. Calibration 
# is set to TIRF microscope settings.
# Adapted from full scripting example from https://imagej.net/plugins/trackmate/scripting
# Author: Pia Mach

#----------------------
# Required update sites
#----------------------

# ImageJ: https://update.imagej.net/
# Fiji: https://update.imagej.net/
# Java-8: https://sites.imagej.net/Java-8/
# BIG-EPFL: https://sites.imagej.net/BIG-EPFL/
# CSBDeep: https://sites.imagej.net/CSBDeep/
# IJPB-Plugins: https://sites.imagej.net/IJPB-plugins/
# ImageScience: https://sites.imagej.net/ImageScience/
# StarDist: https://sites.imagej.net/StarDist/
# TrackMate-StarDist: https://sites.imagej.net/TrackMate-StarDist/

#---------------
# Import modules
#---------------

from java.io import File
import sys
import csv
import os

from ij import IJ
from ij import WindowManager
from ij.io import FileSaver

from fiji.plugin.trackmate.io import TmXmlWriter
from fiji.plugin.trackmate import Model
from fiji.plugin.trackmate import Settings
from fiji.plugin.trackmate import TrackMate
from fiji.plugin.trackmate import SelectionModel
from fiji.plugin.trackmate import Logger
from fiji.plugin.trackmate.tracking import LAPUtils
from fiji.plugin.trackmate.tracking.sparselap import SparseLAPTrackerFactory
from fiji.plugin.trackmate.gui.displaysettings import DisplaySettingsIO
from fiji.plugin.trackmate.stardist import StarDistDetectorFactory
import fiji.plugin.trackmate.visualization.hyperstack.HyperStackDisplayer as HyperStackDisplayer
from fiji.plugin.trackmate.providers import DetectorProvider
from fiji.plugin.trackmate.providers import TrackerProvider
from fiji.plugin.trackmate.providers import SpotAnalyzerProvider
from fiji.plugin.trackmate.providers import EdgeAnalyzerProvider
from fiji.plugin.trackmate.providers import TrackAnalyzerProvider
from fiji.plugin.trackmate.action import LabelImgExporter

# We have to do the following to avoid errors with UTF8 chars generated in 
# TrackMate that will mess with our Fiji Jython.
reload(sys)
sys.setdefaultencoding('utf-8')


#-----------------------------
# Get currently selected image
#-----------------------------

imp = WindowManager.getCurrentImage()
info = imp.getOriginalFileInfo()
basedir = str(info.directory)
ori_name = imp.getTitle()
ori_name = ori_name.replace(".tiff","")

#----------------------------------------------
# Set image calibration, Resize the image, Save
#----------------------------------------------

#imp.getCalibration().setXUnit("px");
#IJ.run(imp, "Properties...", "channels=1 slices=1 frames=100 pixel_width=1 pixel_height=1 voxel_depth=1")
imp = imp.resize(128,128, "none")
imp.show()
imp.setTitle(ori_name + ".scaled.tiff")

imp_scaled_path = File(basedir, imp.getTitle())
imp_scaled = FileSaver(imp)
imp_scaled.saveAsTiff(str(imp_scaled_path))
imp = WindowManager.getCurrentImage()

#----------------------------
# Create the model object now
#----------------------------

# Some of the parameters we configure below need to have
# a reference to the model at creation. So we create an
# empty model now.

model = Model()

# Send all messages to ImageJ log window.
model.setLogger(Logger.IJ_LOGGER)



#------------------------
# Prepare settings object
#------------------------

settings = Settings(imp)

# Configure detector - We use the Strings for the keys
settings.detectorFactory = StarDistDetectorFactory()
settings.detectorSettings = {
    'TARGET_CHANNEL' : 1,
} 

# Configure spot filters - Classical filter on quality
#filter1 = FeatureFilter('QUALITY', 30, True)
#settings.addSpotFilter(filter1)

# Configure tracker - We want to allow merges and fusions
settings.trackerFactory = SparseLAPTrackerFactory()
settings.trackerSettings = LAPUtils.getDefaultLAPSettingsMap() # almost good enough
settings.trackerSettings['LINKING_MAX_DISTANCE'] = float(15)
settings.trackerSettings['ALLOW_GAP_CLOSING'] = True
settings.trackerSettings['ALLOW_TRACK_SPLITTING'] = False
settings.trackerSettings['ALLOW_TRACK_MERGING'] = False

# overwrite tracking parameters
settings.trackerSettings['LINKING_MAX_DISTANCE'] = float(link_dist)#100
settings.trackerSettings['ALLOW_TRACK_SPLITTING'] = False
settings.trackerSettings['ALLOW_TRACK_MERGING'] = False
settings.trackerSettings['ALLOW_GAP_CLOSING'] = True
settings.trackerSettings['GAP_CLOSING_MAX_DISTANCE'] = float(gap_closing_dist)#0
settings.trackerSettings['MAX_FRAME_GAP'] = int(ngap_max)#0

# Add ALL the feature analyzers known to TrackMate. They will 
# yield numerical features for the results, such as speed, mean intensity etc.
settings.addAllAnalyzers()


#-------------------
# Instantiate plugin
#-------------------

trackmate = TrackMate(model, settings)

#--------
# Process
#--------

ok = trackmate.checkInput()
if not ok:
    sys.exit(str(trackmate.getErrorMessage()))

ok = trackmate.process()
if not ok:
    sys.exit(str(trackmate.getErrorMessage()))

#----------------
# Display results
#----------------

# A selection.
selectionModel = SelectionModel( model )

# Read the default display settings.
ds = DisplaySettingsIO.readUserDefault()
displayer =  HyperStackDisplayer(model, selectionModel, imp, ds)
displayer.render()
displayer.refresh()

# Echo results with the logger we set at start:
model.getLogger().log( str( model ) )


#---------------------------------
# Export XML file with cell tracks
#---------------------------------

#outFile_name = imp.getTitle() + ".tracks.xml" 
#outFile = File(basedir, outFile_name)

#writer = TmXmlWriter(outFile) #a File path object
#writer.appendModel( trackmate.getModel() ) #trackmate instantiate like this before trackmate = TrackMate(model, settings)
#writer.appendSettings( trackmate.getSettings() )
#writer.writeToFile()

#---------------------------
# Export all spot statistics
#---------------------------

# Iterate over all the tracks that are visible.

spots = [['track', 'x', 'y', 'frame']]
#remove/add "z" in array if no 3D tracking

# Export as csv
# Get spot coordinate and id
for id in model.getTrackModel().trackIDs(True):
   
       
    track = model.getTrackModel().trackSpots(id)
    for spot in track:
        sid = spot.ID()
        # Fetch spot features directly from spot. 
        x=spot.getFeature('POSITION_X')
        y=spot.getFeature('POSITION_Y')
        #z=spot.getFeature('POSITION_Z') #delete this line, if no 3D tracking
        t=spot.getFeature('POSITION_T')
       
    	spots.append([id, x, y, t ])
    	#remove/add z in array if no 3D tracking
#write output
with open(basedir + "/" + ori_name + ".CellID-tracks.csv", "wb") as f:
    wr = csv.writer(f)
    for row in spots:
        wr.writerow(row)
        
#------------------
# Save label image
#------------------

exportSpotsAsDots = False
exportTracksOnly = False
exportSpotIDsAsLabels = False
lblImg = LabelImgExporter.createLabelImagePlus(trackmate, exportSpotsAsDots, exportTracksOnly, exportSpotIDsAsLabels)
lblImg = lblImg.resize(512,512, "none")
lblImg.show()
outImage_name = ori_name + ".label-image.tiff"
outImage_path = File(basedir, outImage_name)
outImage = FileSaver(lblImg)
outImage.saveAsTiff(str(outImage_path))
