# This script is associated with a specific msem experiment.
# The msem experiment comprises 16 "shift".
# For each "shift", a set of 7 mFOVs is acquired:
#   *  *
# *  *  *
#   *  *
# A mFOV is made of 61 sFOVs.
# There is a translation offset in the x axis between shift number 1 and shift number 2.
# There is the same translation offset between shift n and shift (n+1)
# The translation offset is 1/16 of the width of an sFOV, which is about 1 micrometer.
# The translation offset between shift 0 and shift 8 is 8/16 = 1/2 of the width of a sFOV
# As a first approximation, it could be assumed that the right half of a sFOV
# does not have any distortion. This second half could be taken as a ground truth
# to compute the scan correction

# To find the scan distortion, a vertical 10 pixel wide band from the translated
# image is fit to the non-translated image.

from mpicbg.imglib.image import ImagePlusAdapter
from mpicbg.imglib.algorithm.correlation import CrossCorrelation

from ij import IJ, ImagePlus, ImageStack
from ij.gui import PointRoi, Roi
from ij.plugin import MontageMaker
from ij.plugin.frame import RoiManager
from ij.measure import CurveFitter
from ij.process import ColorProcessor

from bdv.ij import ApplyBigwarpPlugin
from bdv.viewer import Interpolation
from bigwarp.landmarks import LandmarkTableModel

from java.io import File

from java.lang import Math

from net.imglib2.img.array import ArrayImgFactory
from net.imglib2.type.numeric.real import FloatType
from net.imglib2.type.numeric.complex import ComplexFloatType
from net.imglib2.view import Views
from java.util.concurrent import Executors
from java.lang import Runtime
from java.awt import Rectangle
from net.imglib2.img.display.imagej import ImageJFunctions as IL

from net.imglib2.algorithm.fft2 import FFTMethods
from net.imglib2.algorithm.fft2 import FFT
from net.imglib2.algorithm.phasecorrelation import PhaseCorrelation2
from net.imglib2.algorithm.phasecorrelation import PhaseCorrelation2Util
from jarray import array
from net.imglib2 import FinalInterval
from ij.gui import Plot

import os

def exp_fit(x, a, b, c):
    return a*Math.exp(-b*x) + c

def poly2_fit(x, a, b, c):
    return a + b*x + c*x*x

def poly3_fit(x, a, b, c, d):
    x2 = x*x
    return a + b*x + c*x*x + d *x2*x

def crop(im,roi):
	ip = im.getProcessor()
	ip.setRoi(roi)
	im = ImagePlus(im.getTitle() + '_Cropped', ip.crop())
	return im

def getCC(im1,im2):
	im1, im2 = map(ImagePlusAdapter.wrap, [im1, im2])
	cc = CrossCorrelation(im1, im2)
	cc.process()
	return cc.getR()

def getShiftFromViews(v1, v2):
    # Thread pool
    exe = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors())
    try:
        # PCM: phase correlation matrix
        pcm = PhaseCorrelation2.calculatePCM(v1,
	                                       v2,
	                                       ArrayImgFactory(FloatType()),
	                                       FloatType(),
	                                       ArrayImgFactory(ComplexFloatType()),
	                                       ComplexFloatType(),
	                                       exe)
        # Minimum image overlap to consider, in pixels
        minOverlap = v1.dimension(0) / 10
        # Returns an instance of PhaseCorrelationPeak2
        peak = PhaseCorrelation2.getShift(pcm, v1, v2, nHighestPeaks,
	                                    minOverlap, True, True, exe)
    except Exception, e:
        print e
    finally:
        exe.shutdown()

    # Register images using the translation (the "shift")
    spshift = peak.getSubpixelShift()
    return spshift.getFloatPosition(0), spshift.getFloatPosition(1)

def getShiftFromImps(imp1, imp2):
    v1 = getViewFromImp(imp1)
    v2 = getViewFromImp(imp2)
    return getShiftFromViews(v1, v2)

def getViewFromImp(imp, r = None):
    # r is a java.awt.rectangle
    im = IL.wrapByte(imp)
    if r is None:
        r = Rectangle(0, 0, imp.getWidth(), imp.getHeight())
    v = Views.zeroMin(Views.interval(im, [r.x, r.y],
                        [r.x + r.width -1, r.y + r.height -1]))
    return v

def getViewFromImglib2Im(img, r):
    v = Views.zeroMin(Views.interval(img, [r.x, r.y],
                        [r.x + r.width -1, r.y + r.height -1]))
    return v

def getFFTParamsFromImps(imp1, imp2, r1 = None, r2 = None):
    if r1:
        v1 = getViewFromImp(
            imp1,
            r1)
    else:
        v1 = IL.wrapByte(imp1)
    if r2:
        v2 = getViewFromImp(
            imp2,
            r2)
    else:
        v2 = IL.wrapByte(imp2)

    extension = array(v1.numDimensions() * [10], 'i')
    extSize = PhaseCorrelation2Util.getExtendedSize(v1, v2, extension)
    paddedDimensions = array(extSize.numDimensions() * [0], 'l')
    fftSize = array(extSize.numDimensions() * [0], 'l')
    return extension, extSize, paddedDimensions, fftSize

def getFFTFromView(v, extension, extSize, paddedDimensions, fftSize):
    FFTMethods.dimensionsRealToComplexFast(extSize, paddedDimensions, fftSize)
    fft = ArrayImgFactory(ComplexFloatType()).create(fftSize, ComplexFloatType())
    FFT.realToComplex(Views.interval(PhaseCorrelation2Util.extendImageByFactor(v, extension),
            FFTMethods.paddingIntervalCentered(v, FinalInterval(paddedDimensions))), fft, exe)
    return fft

def getShiftFromFFTs(fft1, fft2, v1, v2, minOverlap, nHighestPeaks):
    pcm = PhaseCorrelation2.calculatePCMInPlace(
        fft1,
        fft2,
        ArrayImgFactory(
            FloatType()),
            FloatType(),
            exe)
    peak = PhaseCorrelation2.getShift(pcm, v1, v2, nHighestPeaks,
                                    minOverlap, True, True, exe)
    spshift = peak.getSubpixelShift()
    if spshift is not None:
        return spshift.getFloatPosition(0), spshift.getFloatPosition(1)
    else:
        IJ.log('There is a peak.getSubpixelShift issue. sFOV ' + str(sFOV) + ' s ' + str(s))
        return None

# order of the 61 beams, line by line left to right. Used to display the topological map of the beams
HEXAGONS = [46,45,44,43,42,47,26,25,24,23,41,48,27,12,11,10,22,40,49,28,13,4,3,9,21,39,50,29,14,5,1,2,8,20,38,51,30,15,6,7,19,37,61,52,31,16,17,18,36,60,53,32,33,34,35,59,54,55,56,57,58]
HEXAGON_SPACINGS = [4,1,1,1,1,7,1,1,1,1,1,5,1,1,1,1,1,1,3,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,3,1,1,1,1,1,1,5,1,1,1,1,1,7,1,1,1,1,4]

###################################
# select the root of the experiment
# # 8 nm pixel; 1 um sFOV overlap; 400 ns dwell time 
# root = r'M:\hess\calibration_tests_10_2020\calibration_8nm_1um_400ns_noSC__20201002_12-24-03'

# # 8 nm pixel; 1 um sFOV overlap; 1600 ns dwell time 
# root = r'M:\hess\calibration_tests_10_2020\calibration_8nm_1um_1600ns_noSC_20201002_11-17-51'

# # 4 nm pixel; 0.5 um sFOV overlap; 400 ns dwell time 
# root = r'M:\hess\calibration_tests_10_2020\calibration_4nm_05um_400ns_noSC_20201002_12-48-40'

# 4 nm pixel; 0.5 um sFOV overlap; 1600 ns dwell time 
root =  r'M:\hess\calibration_tests_10_2020\calibration_4nm_05um_1600ns_noSC_20201002_13-22-16'
###################################

# names of the folders containing the different "shift" experiments
shiftFolderNames = [str(i+1).zfill(3) + '_R_' + str(i+1) for i in range(16)]

# Number of phase correlation peaks to check with cross-correlation
nHighestPeaks = 10

# Each shift contains 7 mFOVs. Any of these mFOVs can be used.
mFOV = 0

# which of the 16 shifts should be compared with shift 0
shift_n = 9

plot_images = []

# get the ROI manager
roi_manager = RoiManager.getInstance()
if roi_manager == None:
    roi_manager = RoiManager()

#
all_pc = []

x_to_fit = []
y_to_fit = []

# width of the sliding vertical window
vertical_strip_width = 50

# im_coordinates_all_spshifts is the list of paths of the text files "full_image_coordinates.txt" that contain the
# coordinates of all sFOVs in each shift experiment.
# im_coordinates_all_spshifts[8] contains the path and x,y coordinates of the 427 sFOVs (61 sFOV * 7 mFOV)
# of shift number 8
im_coordinates_all_spshifts = [os.path.join(root, shiftFolderName, 'full_image_coordinates.txt')
    for shiftFolderName in shiftFolderNames]

# executor service
exe = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors())

# for sFOV in HEXAGONS[0:10]:
# for sFOV in [23]:

# path of the .txt file that stores the scan correction fits for all beams
scan_distortion_measurement_path = os.path.join(
    root,
    ('scan_distortion_measurements_shift_'
        + str(shift_n).zfill(2)
        + '_mFOV_' + str(mFOV)
        + '.txt'))
# path of the overview montage that shows all beam corrections
fit_montage_path = os.path.join(
    root,
    'fit_montage.tif')

with open(scan_distortion_measurement_path, 'w') as g:
    for sFOV in HEXAGONS:
        sFOV = sFOV-1

        # finding the path of a given sFOV by checking its filename
        with open(im_coordinates_all_spshifts[0], 'r') as f:
            lines = f.readlines()
            path_tail_1 = [x.split('\t')[0]
                for x in lines
                if (str(mFOV+1).zfill(6) + '_' + str(sFOV + 1).zfill(3)) in x][0]

        # finding the path of a given sFOV by checking its filename
        with open(im_coordinates_all_spshifts[shift_n], 'r') as f:
            lines = f.readlines()
            path_tail_2 = [x.split('\t')[0]
                for x in lines
                if (str(mFOV+1).zfill(6) + '_' + str(sFOV + 1).zfill(3)) in x][0]

        # paths of the same sFOV in the first and in the second shift
        im_path_1 = os.path.join(root, shiftFolderNames[0], path_tail_1)
        im_path_2 = os.path.join(root, shiftFolderNames[shift_n], path_tail_2)

        im_1 = IJ.openImage(im_path_1)
        im_2 = IJ.openImage(im_path_2)

        # im_1.show()
        # im_2.show()

        # use the last 10% on the right of im_1 in order to align im_1 to im_2
        start_of_last_10_pct_of_im_1 = Math.floor(0.9 * (im_1.getWidth()))

        # extracting the 10% right vertical band of im_1
        #           _
        # *********|*|
        # *********|*|
        # *********|*|
        # *********|*|
        # *********|*|
        # *********|*|
        #           -

        last_10_pct_of_im_1 = crop(
            im_1,
            Roi(
                start_of_last_10_pct_of_im_1,
                0,
                im_1.getWidth() - start_of_last_10_pct_of_im_1 + 1,
                im_1.getHeight()))

        # last_10_pct_of_im_1.show()

        # calculate the translation offset between im_1 and im_2
        xy_subpixel_shift = getShiftFromImps(last_10_pct_of_im_1, im_2)
        end_of_im_1_in_im_2 = -xy_subpixel_shift[0] + last_10_pct_of_im_1.getWidth()
        IJ.log('end_of_im_1_in_im_2 - ' + str(end_of_im_1_in_im_2))

        img_1 = IL.wrapByte(im_1) #im is an ImagePlus, img is a ImgLib2 image
        img_2 = IL.wrapByte(im_2)

        # get the fft parameters only once at the beginning because they will be te same for all calculations
        extension, extSize, paddedDimensions, fftSize = getFFTParamsFromImps(
            im_1,
            im_2,
            r1 = Rectangle(0, 0, 3*vertical_strip_width, im_2.getHeight()),
            r2 = Rectangle(0, 0, vertical_strip_width, im_2.getHeight()))

        # initializing list of subpixelshifts
        spshifts = []

        # sliding the small vertical band (which will be called view v2) pixel by pixel
        # the start of the small vertical band in im_2 is start_of_sliding_band_in_im_2, renamed as "s"
        s_range = range(int(Math.ceil(end_of_im_1_in_im_2 - vertical_strip_width)))

        for start_of_sliding_band_in_im_2 in s_range:
        # for start_of_sliding_band_in_im_2 in range(30):
            s = start_of_sliding_band_in_im_2
            # extract view v1. Instead of looking for where the narrow band v2 fits in the entire im_1,
            # we extract only a narrow band in im_1. We know that v2 will be found in v1.
            # We take the narrow band 3 times wider than v2.

            # the following is clear when looking at the schematic
            start_of_v1_in_im_1 = im_1.getWidth() - (int(round(end_of_im_1_in_im_2)) - s + vertical_strip_width)

            r1 = Rectangle(
                start_of_v1_in_im_1,
                0,
                3*vertical_strip_width,
                im_2.getHeight())
            v1 = getViewFromImglib2Im(img_1, r1)
            # if s == 29:
                # crop(im_1, r1).show()
            minOverlap = v1.dimension(0) / 10

            # compute fft_1 of view v1
            fft_1 = getFFTFromView(v1, extension, extSize, paddedDimensions, fftSize)

            if s%50 == 0:
                IJ.log('s-' + str(s).zfill(4) + ' sFOV ' + str(sFOV))

            # extracting view v2 and computing its fft
            r2 = Rectangle(s, 0, vertical_strip_width, im_2.getHeight())
            v2 = getViewFromImglib2Im(img_2, r2)
            # if s == 29:
                # crop(im_2, r2).show()
            fft_2 = getFFTFromView(v2, extension, extSize, paddedDimensions, fftSize)

            # calculating the spshift between v1 and v2
            # spshift stands for subpixelshift (differs from the "shift" experiments)
            spshift = getShiftFromFFTs(fft_1, fft_2, v1, v2, minOverlap, nHighestPeaks)
            if spshift is not None:
                spshifts.append([s, spshift[0]])

        # # plot the scan distortion of the current sFOV
        # plot = Plot('Scan distortion', 'spshift', 'offset')
        # plot.add(
            # 'circle',
            # [a[0] for a in spshifts],
            # [vertical_strip_width - a[1] for a in spshifts])
        # plot.show()

        # stack the scan correction measurements in one list to later
        # fit a single scan correction for all beams.
        # (even though a correction for each of the 61 beams is probably needed)
        x_to_fit = x_to_fit + [a[0] for a in spshifts]
        y_to_fit = y_to_fit + [vertical_strip_width - a[1] for a in spshifts]

        cv = CurveFitter(
            [a[0] for a in spshifts],
            [vertical_strip_width - a[1] for a in spshifts])

        # fit the distortion measurements
        cv.doFit(CurveFitter.EXP_WITH_OFFSET)
        # cv.doFit(CurveFitter.EXPONENTIAL)
        # cv.doFit(CurveFitter.POLY2)
        # cv.doFit(CurveFitter.POLY3)

        plot = cv.getPlot()
        # plot.show()
        plot_images.append(plot.getImagePlus())
        IJ.log('sFOV ' + str(sFOV))
        IJ.log('fitGoodness ' + str(cv.getFitGoodness()))
        IJ.log('formula ' + str(cv.getFormula()))
        fit_params = cv.getParams()
        IJ.log('fit_params' + str(fit_params))      
        
        # write the fit results to file
        g.write(
            'sFOV\t' + str(sFOV) + '\t'
            + 'a\t' + str(fit_params[0]) + '\t'
            + 'b\t' + str(fit_params[1]) + '\t'
            + 'c\t' + str(fit_params[2]) + '\t'
            + 'formula\t' + str(cv.getFormula()) + '\t'
            + 'fitGoodness\t' + str(cv.getFitGoodness()) + '\n')


        # # # # # The following commented code applies a scan correction using bigwarp.
        # # # # # A grid of landmarks is generated, saved into bigwarp format, then applied.
        # # # # # It is slow and needs to be optimized.

        # # # # create mesh
        # # # im_h = im_1.getHeight()
        # # # im_w = im_1.getWidth()

        # # # # mesh
        # # # mesh_x_n = 50 # number of points in the mesh on x-axis
        # # # mesh_x = range(0, im_w, im_w/mesh_x_n)

        # # # mesh_y_n = 30
        # # # mesh_y = range(0, im_h, im_h/mesh_y_n)

        # # # # normal mesh for im_1
        # # # landmarks_im_1 = [
            # # # [x,y]
            # # # for x in mesh_x
            # # # for y in mesh_y]
        # # # # warped mesh for im_2
        # # # landmarks_im_2 = [
            # # # # [x - cv.f(x - mean_x_alignment_band_im2), y] # applying the min_im2_features_x offset because the fit was calculated with this offset
            # # # [x - exp_fit(x - mean_x_alignment_band_im2,
                # # # 0.006080825459861896,
                # # # 0.005848866230539375,
                # # # -0.23867705612301265),
            # # # y]
            # # # for x in mesh_x
            # # # for y in mesh_y]
        # # # roi2pm = PointRoi()

        # # # # for landmark in landmarks_im_2:
            # # # # roi2pm.addPoint(*landmark)
        # # # # roi_manager.addRoi(roi2pm)

        # # # # save landmarks in bigwarp format
        # # # landmarks_path = os.path.join(root, 'landmarks.csv')
        # # # with open(landmarks_path, 'w') as f:
            # # # for id, (l1,l2) in enumerate(zip(landmarks_im_1, landmarks_im_2)):
                # # # f.write(','.join([
                    # # # '"Pt-' + str(id) + '"',
                    # # # '"true"',
                    # # # '"' + str(l1[0]) + '"',
                    # # # '"' + str(l1[1]) + '"',
                    # # # '"' + str(l2[0]) + '"',
                    # # # '"' + str(l2[1]) + '"']))
                # # # f.write('\n')
        # # # ltm = LandmarkTableModel(2)
        # # # ltm.load(File(landmarks_path))

        # # # im_2_warped = ApplyBigwarpPlugin.apply(
                # # # im_2, im_2, ltm,
                # # # 'Target', '', 'Target',
                # # # None, None, None,
                # # # Interpolation.NLINEAR, False, 16)
        # # # # im_2_warped.show()

        # # # im_1_cropped = crop(
            # # # im_1,
            # # # Roi(alignment_offset,0,crop_x,im_h))
        # # # im_2_cropped = crop(
            # # # im_2_warped,
            # # # Roi(0,0,crop_x,im_h))

        # # # # translating to account for y-axis translation
        # # # IJ.run(
            # # # im_2_cropped,
            # # # 'Translate...',
            # # # 'x=0 y=' + str(alignment_offset_y) + ' interpolation=Bilinear')

        # # # #crop a last time in y to account for the y-translation and avoid black region
        # # # im_1_cropped = crop(
            # # # im_1_cropped,
            # # # Roi(0,25,im_1_cropped.getWidth(),im_1_cropped.getHeight()-50))
        # # # im_2_cropped = crop(
            # # # im_2_cropped,
            # # # Roi(0,25,im_2_cropped.getWidth(),im_2_cropped.getHeight()-50))

        # # # # im_1_cropped.show()
        # # # # im_2_cropped.show()

        # # # stack_overlap = ImageStack(
            # # # im_2_cropped.getWidth(),
            # # # im_2_cropped.getHeight())
        # # # stack_overlap.addSlice(im_1_cropped.getProcessor())
        # # # stack_overlap.addSlice(im_2_cropped.getProcessor())
        # # # im_overlap = ImagePlus(
            # # # 'sFOV_' + str(sFOV).zfill(2),
            # # # stack_overlap)
        # # # # im_overlap.show()

        # # # cc = getCC(
            # # # im_1_cropped,
            # # # im_2_cropped)
        # # # print sFOV, 'cc', cc
        # # # all_pc.append(cc)

        # # # # End of applying scan correction


# # # global scan correction for all beams
# # for fit_model in [
    # # CurveFitter.EXP_WITH_OFFSET,
    # # CurveFitter.POLY2,
    # # CurveFitter.POLY3]:

    # # cv = CurveFitter(
        # # x_to_fit,
        # # y_to_fit)

    # # cv.doFit(fit_model)
    # # plot = cv.getPlot()
    # # plot.show()
    # # # plot_images.append(plot.getImagePlus())
    # # # print 'sFOV', sFOV
    # # print 'fit_model', fit_model
    # # print 'fit', cv.getFit()
    # # print 'fitGoodness', cv.getFitGoodness()
    # # print 'formula', cv.getFormula()
    # # fit_params = cv.getParams()
    # # print 'fit_params', fit_params

# print all_pc
# print 'mean_cc', sum(all_pc)/len(all_pc)

# Create the 2D visualization of the fit plots
stack_w, stack_h = plot_images[0].getWidth(), plot_images[0].getHeight()
stack_plot = ImageStack(stack_w, stack_h)

# add all the plots in a stack
for id, plot_image in enumerate(plot_images):
    for i in range(HEXAGON_SPACINGS[id]):
        stack_plot.addSlice(
            'plot_' + str(id).zfill(2),
            ImagePlus('empty', ColorProcessor(stack_w, stack_h)).getProcessor())
    stack_plot.addSlice('plot_' + str(id).zfill(2), plot_image.getProcessor())

imp = ImagePlus('Plot_stack', stack_plot)

montageMaker = MontageMaker()
montage = montageMaker.makeMontage2(
	imp, 
	17, 9,
    1, 1,
    imp.getNSlices(), 1, 3, False)

IJ.save(
    montage,
    fit_montage_path)

montage.show()
IJ.log('Done')