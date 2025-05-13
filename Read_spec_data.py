'''
Reading in data from HST is a bit of a nuisance, and it takes
up a lot of lines of code in the coadd/rebin script.  So, should
hide it away in this separate file.  Ideally, reading the data should
be the only free parameter in this process.
'''

import pandas as pd
import Cut_Edge_Pix_TVM
import SpecCuts_HSLA
import numpy as np
from astropy.io import fits
import glob

def read_data(Identifier, path, data_origin, z):
    if data_origin=="FOS":
        return read_fos(Identifier, path, z)
    elif data_origin=="STIS":
        return read_stis(Identifier, path, z)
    elif data_origin=="COS":
        return read_cos(Identifier, path, z)
    elif data_origin=="HSLA":
        return read_hsla(Identifier, path, z)
    elif data_origin=="SDSS-RM":
        return read_sdssrm(Identifier, path, z)
    print("data_origin not recongnized")

def read_sdssrm(Identifier, path, z):
    fn_list = glob.glob(path+"%s/*.fits"%Identifier)
    array_lens = []
    for i in range(len(fn_list)):
        wavelength = 10.**fits.open(fn_list[i])[1].data["LOGLAM"]
        array_lens.append(len(wavelength))
    array_len = max(array_lens)

    waves  = np.zeros((len(fn_list), array_len))
    fluxes = np.zeros((len(fn_list), array_len))
    errs   = np.zeros((len(fn_list), array_len))
    masks  = np.zeros((len(fn_list), array_len))

    for i in range(len(fn_list)):
        hdu = fits.open(fn_list[i])
        loglam = hdu[1].data["LOGLAM"]
        wave   = 10.**loglam
        flux   = hdu[1].data["FLUX"]
        err    = 1. / np.sqrt(hdu[1].data["IVAR"])
        mask   = hdu[1].data["AND_MASK"]
        #save in full arrays
        #Cut_Edge_Pix_TVM breaks for some noisy RM spectra
        #waves[i,:len(wave)]  = Cut_Edge_Pix_TVM.Cut_Edge_Pix(mask, wave, flux, err, wave, array_len, "wavelength", False, z)
        #fluxes[i,:len(wave)] = Cut_Edge_Pix_TVM.Cut_Edge_Pix(mask, wave, flux, err, flux, array_len, "wavelength", False, z)
        #errs[i,:len(wave)]   = Cut_Edge_Pix_TVM.Cut_Edge_Pix(mask, wave, flux, err, err, array_len, "wavelength", False, z)
        #masks[i,:len(wave)]  = Cut_Edge_Pix_TVM.Cut_Edge_Pix(mask, wave, flux, err, mask, array_len, "wavelength", False, z)
        waves[i,:len(wave)]  = wave
        fluxes[i,:len(wave)] = flux
        errs[i,:len(wave)]   = err
        masks[i,:len(wave)]  = mask

    #should include Cut_Edge_Pix_TVM here?

    return waves, fluxes, errs, masks

def read_hsla(Identifier, path, z):
    spec = fits.open(path+"original/%s.fits"%Identifier)
    coadd_wave     = spec[1].data["WAVE"]
    coadd_flux     = spec[1].data["FLUX"]
    coadd_errs     = spec[1].data["ERROR"]
    coadd_mask     = np.zeros(len(coadd_wave))

    #Manually cut red end - big problem for HSLA co-adds
    obj_name = "_".join(Identifier.split("_")[:-1])
    #print(obj_name)
    if obj_name in SpecCuts_HSLA.RedEdges:
        lambdaend = SpecCuts_HSLA.RedEdges[obj_name]
        indstop  = np.argmin( np.abs((coadd_wave/(1+z))-lambdaend) )
        coadd_wave = coadd_wave[:indstop]
        coadd_flux = coadd_flux[:indstop]
        coadd_errs = coadd_errs[:indstop]
        coadd_mask = coadd_mask[:indstop]

    coadd_wave   = Cut_Edge_Pix_TVM.Cut_Edge_Pix(np.zeros(len(coadd_wave)), coadd_wave, coadd_flux, coadd_errs, coadd_wave.copy(), len(coadd_wave), "wavelength", False, z, "%s"%(Identifier), "HSLA")
    coadd_flux   = Cut_Edge_Pix_TVM.Cut_Edge_Pix(np.zeros(len(coadd_wave)), coadd_wave, coadd_flux, coadd_errs, coadd_flux.copy(), len(coadd_flux), "flux", False, z, "%s"%(Identifier), "HSLA")
    #HSLA saves empty values as zero; change to nan
    coadd_flux[coadd_flux==0] = np.nan
    coadd_errs   = Cut_Edge_Pix_TVM.Cut_Edge_Pix(np.zeros(len(coadd_wave)), coadd_wave, coadd_flux, coadd_errs, coadd_errs.copy(), len(coadd_errs), "flux error", False, z, "%s"%(Identifier), "HSLA")
    coadd_mask   = Cut_Edge_Pix_TVM.Cut_Edge_Pix(np.zeros(len(coadd_wave)), coadd_wave, coadd_flux, coadd_errs, coadd_mask.copy(), len(coadd_mask), "masks", False, z, "%s"%(Identifier), "HSLA")

    return np.array([coadd_wave]), np.array([coadd_flux]), np.array([coadd_errs]), np.array([coadd_mask])

def read_cos(Identifier, path, z):
    #get observation details
    try:
        obs_details = pd.read_csv(path+"%s/all_exposures.txt" %(Identifier), sep="\s+")
        gratings    = obs_details["Grating"].values
        spec_names  = obs_details["Rootname"].values
    except FileNotFoundError:
        obs_details = pd.read_csv(path+"%s/NecessaryParams.csv" %(Identifier))
        gratings    = obs_details["filters"].values
        spec_names  = obs_details["obs_id"].values

    #take max wave size for initializing below
    array_sizes = []
    for i in range(len(spec_names)):
        try:
            data = fits.open(path+'%s/%s/%s_x1d.fits' %(Identifier,spec_names[i],spec_names[i])) #might want try/except to handle files with directories
        except FileNotFoundError:
            try:
                data = fits.open(path+'%s/%s.fits' %(Identifier,spec_names[i]))
            except FileNotFoundError:
                data = fits.open(path+'%s/Data/%s_x1d.fits' %(Identifier,spec_names[i]))
        data = data[1].data
        if gratings[i]=='E140M':
            #FIXME: Ignore echelle gratings for now
            array_sizes.append(data.size*1024) #more than one, force to be 1024 size each
        else:
            #print(len(data['Wavelength'][0]))
            array_sizes.append(len(data['Wavelength'][0]))
    array_len=max(array_sizes)

    #Initialize - 0th axis for each pixel, 1st axis has number of observations
    waves     = np.zeros((len(spec_names), array_len))
    fluxes    = np.zeros((len(spec_names), array_len))
    flux_errs = np.zeros((len(spec_names), array_len))
    masks     = np.zeros((len(spec_names), array_len))

    for i in range(len(spec_names)):
        #Designate a list to keep any low S/N images in.
        Bad_list=[]
        try:
            data = fits.open(path+'%s/%s/%s_x1d.fits' %(Identifier,spec_names[i],spec_names[i]))[1].data
        except FileNotFoundError:
            try:
                data = fits.open(path+'%s/%s.fits' %(Identifier,spec_names[i]))[1].data
            except FileNotFoundError:
                data = fits.open(path+'%s/Data/%s_x1d.fits' %(Identifier,spec_names[i]))[1].data
        #print(data.size)
        #Note that for the echelle gratings there are lot of little windows, i.e.-44 extensions for PG1444... :O Can concatenate, and fill in with zeros using the methods I implement in the COS code.

        #Need to ID whether or not the echelle grating is the only one or not:
        #Even if that's there is a clause since not all Echelle observations were the same number of windows!
        if gratings[i]=="E140M":
            wavelength=[]
            flux=[]
            fluxerr=[]
            DQ=[]
            for t in np.arange(0,data.size,1):
                wavelength = np.append(wavelength, data['WAVELENGTH'][t])
                flux       = np.append(flux, data['FLUX'][t])
                fluxerr    = np.append(fluxerr, data['ERROR'][t])
                DQ         = np.append(DQ, data['DQ'][t])
        else:
            #should loop through here?
            wavelength = data['WAVELENGTH'][0]
            flux       = data['FLUX'][0]
            fluxerr    = data['ERROR'][0]
            DQ         = data['DQ'][0]



        '''
        Step 2: Masks bad pixels -- both as classified by HST and those
                in overlapping skyline regions.

                Also cut edge pixels with consectutive
                1) Negative waves (why does this happen??) #Not in COS
                2) Negative fluxes #Not in COS
                3) Masked pixels
                Note that this is relevant as the edges of the spectra
                will be lower S/N.
        '''
        #Mask sky lines and given bad pixels from HST
        flux_wmask     = flux.copy()
        err_wmask      = fluxerr.copy()
        sel            = ((wavelength >= 1215.0) & (wavelength <= 1216.0)) #For COS the optimal values to exclude are 1215-1216 angstroms for Lyman alpha.
        #flux_wmask[(DQ!=0)|sel] = np.nan #FIXME: input correct masking
        #err_wmask[(DQ!=0)|sel] = 0.0
        #print(masks.shape, err_wmask.shape)
        masks[i,:][(err_wmask == 0.0)] = 1
        """
        plt.plot(wavelength,flux, zorder=1)
        plt.scatter(wavelength[DQ!=0],flux[DQ!=0], color="r", zorder=2)
        plt.show()

        plt.plot(wavelength/(1+z), flux)
        plt.scatter(wavelength[:65]/(1+z), flux[:65], color="r")
        plt.title("Flux before cut")
        plt.show()
        plt.plot(wavelength/(1+z), flux/fluxerr)
        plt.title("SNR before cut")
        plt.show()
        """
        waves[i,:]     = Cut_Edge_Pix_TVM.Cut_Edge_Pix(DQ, wavelength, flux_wmask, err_wmask, \
                                                wavelength, array_len, "wavelength", False, z, "%s - %s"%(Identifier,spec_names[i]), "COS")
        fluxes[i,:]    = Cut_Edge_Pix_TVM.Cut_Edge_Pix(DQ, wavelength, flux_wmask, err_wmask, \
                                                flux_wmask, array_len, "flux", False, z, "%s - %s"%(Identifier,spec_names[i]), "COS")
        flux_errs[i,:] = Cut_Edge_Pix_TVM.Cut_Edge_Pix(DQ, wavelength, flux_wmask, err_wmask, \
                                                err_wmask, array_len, "flux error", False, z, "%s - %s"%(Identifier,spec_names[i]), "COS")
        masks[i,:]     = Cut_Edge_Pix_TVM.Cut_Edge_Pix(DQ, wavelength, flux_wmask, err_wmask, \
                                                masks[i,:], array_len, "masks", False, z, "%s - %s"%(Identifier,spec_names[i]), "COS")

    return waves, fluxes, flux_errs, masks

def read_stis(Identifier, path, z):
    obs_details = pd.read_csv(path+"%s/NecessaryParams.csv" %(Identifier))
    gratings    = obs_details["filters"].values
    spec_names  = obs_details["obs_id"].values

    array_sizes = []
    for i in range(len(spec_names)):
        try:
            data = fits.open(path+'%s/%s/%s_x1d.fits' %(Identifier,spec_names[i],spec_names[i]))
        except FileNotFoundError:
            data = fits.open(path+'%s/%s/%s_sx1.fits' %(Identifier,spec_names[i],spec_names[i]))
        data = data[1].data
        if gratings[i]=='E140M':
            array_sizes.append(data.size*1024) #more than one, force to be 1024 size each
        else:
            #print(len(data['Wavelength'][0]))
            array_sizes.append(len(data['Wavelength'][0]))
    array_len=max(array_sizes)

    #Initialize - 0th axis for each pixel, 1st axis has a dimension per observation
    waves     = np.zeros((len(spec_names), array_len))
    fluxes    = np.zeros((len(spec_names), array_len))
    flux_errs = np.zeros((len(spec_names), array_len))
    masks     = np.zeros((len(spec_names), array_len))

    for i in range(len(spec_names)):
        try:
            data = fits.open(path+'%s/%s/%s_x1d.fits' %(Identifier,spec_names[i],spec_names[i]))[1].data
        except FileNotFoundError:
            data = fits.open(path+'%s/%s/%s_sx1.fits' %(Identifier,spec_names[i],spec_names[i]))[1].data
        #Need to ID whether or not the echelle grating is the only one or not:
        #Even if that's there is a clause since not all Echelle observations were the same number of windows!
        if gratings[i]=="E140M":
            wavelength=[]
            flux=[]
            fluxerr=[]
            DQ=[]
            for t in np.arange(0,data.size,1):
                wavelength = np.append(wavelength, data['WAVELENGTH'][t])
                flux       = np.append(flux, data['FLUX'][t])
                fluxerr    = np.append(fluxerr, data['ERROR'][t])
                DQ         = np.append(DQ, data['DQ'][t])
        else:
            #should
            wavelength = data['WAVELENGTH'][0]
            flux       = data['FLUX'][0]
            fluxerr    = data['ERROR'][0]
            DQ         = data['DQ'][0]


        #Mask sky lines and given bad pixels from HST
        flux_wmask     = flux.copy()
        err_wmask      = fluxerr.copy()
        sel            = ((wavelength >= 1215.0) & (wavelength <= 1216.0)) #For STIS the optimal values to exclude are 1215-1216 angstroms for Lyman alpha.
        #flux_wmask[(DQ==2**14)|sel] = np.nan
        #err_wmask[(DQ==2**14)|sel] = 0.0
        #print(masks.shape, err_wmask.shape)
        masks[i,:len(err_wmask)][err_wmask==0.] = 1
        waves[i,:len(err_wmask)]     = Cut_Edge_Pix_TVM.Cut_Edge_Pix(DQ, wavelength, flux_wmask, err_wmask, \
                                                wavelength, array_len, "wavelength", False, z, "%s - %s"%(Identifier,spec_names[i]), "STIS")
        fluxes[i,:len(err_wmask)]    = Cut_Edge_Pix_TVM.Cut_Edge_Pix(DQ, wavelength, flux_wmask, err_wmask, \
                                                flux_wmask, array_len, "flux", False, z, "%s - %s"%(Identifier,spec_names[i]), "STIS")
        flux_errs[i,:len(err_wmask)] = Cut_Edge_Pix_TVM.Cut_Edge_Pix(DQ, wavelength, flux_wmask, err_wmask, \
                                                err_wmask, array_len, "flux error", False, z, "%s - %s"%(Identifier,spec_names[i]), "STIS")
        masks[i,:len(err_wmask)]     = Cut_Edge_Pix_TVM.Cut_Edge_Pix(DQ, wavelength, flux_wmask, err_wmask, \
                                                masks[i,:len(err_wmask)], array_len, "masks", False, z, "%s - %s"%(Identifier,spec_names[i]), "STIS")

    return waves, fluxes, flux_errs, masks

def read_fos(Identifier, path, z):

    #Details for specific set of HST observations for a given object
    obs_details = pd.read_csv(path+"%s/NecessaryParams.csv"%Identifier)
    gratings    = obs_details["filters"].values
    spec_names  = obs_details["obs_id"].values

    """
    Some observations are taken with different gratings so the wave
    arrays for different observations are different sizes.  Save the
    length of the longest one.
    """
    array_lens = np.array([], dtype=int)
    for spectrum in spec_names:
        wave = fits.open(path+"%s/%s/%s_c0f.fits" % (Identifier, spectrum, spectrum))[0].data
        array_lens = np.append(array_lens, wave.shape[-1]) #save length of wave array(s)
    array_len = max(array_lens)

    #Initialize - 0th axis for each pixel, 1st axis has a dimension per observation
    waves     = np.zeros((len(spec_names), array_len))
    fluxes    = np.zeros((len(spec_names), array_len))
    flux_errs = np.zeros((len(spec_names), array_len))
    masks     = np.zeros((len(spec_names), array_len))


    def accum_flag(index):
        #Check if data are already accumulated
        return np.mean(flux[index]) > np.mean(flux[index-1]) and np.mean(flux[index-1]) > np.mean(flux[index-2])

    for i in range(len(spec_names)):
        wave = fits.open(path+"%s/%s/%s_c0f.fits" % (Identifier, spec_names[i], spec_names[i]))[0].data

        ind = -1 #check final wavelength array - doesn't really matter for wave, but does for flux below

        #Find the size of the usable wavelength array and reverse if needed
        if len(wave.shape) > 1:
            nzero = (wave[ind]>0.) #don't let zeros mess up condition
            if (wave[ind][nzero][0] > wave[ind][nzero][-1]):
                wavelength = wave[ind][::-1]
            else:
                wavelength = wave[ind][:]
        else:
            nzero = (wave>0.)
            if (wave[nzero][0] > wave[nzero][-1]):
                wavelength = wave[::-1]
            else:
                wavelength = wave[:]
        # Sometimes the wavelength arrays are given in reverse order, the code above reverses ehir order if this is the case. - AP

        #Initialize for this observation - note that needs to be after the above to get sizes correct
        flux    = np.zeros(len(wavelength))
        fluxerr = np.zeros(len(wavelength))
        DQ      = np.zeros(len(wavelength)) #flags

        #Load in arrays for each observation
        obs_flux    = fits.open(path+'%s/%s/%s_c1f.fits' %(Identifier,spec_names[i],spec_names[i]))[0].data
        obs_fluxerr = fits.open(path+'%s/%s/%s_c2f.fits' %(Identifier,spec_names[i],spec_names[i]))[0].data
        obs_DQ      = fits.open(path+'%s/%s/%s_cqf.fits' %(Identifier,spec_names[i],spec_names[i]))[0].data

        #Check if multiple observations in file
        if len(wave.shape) > 1:
            #Check if spectra already accumulated
            if accum_flag(ind):
                if (wave[ind][nzero][0] > wave[ind][nzero][-1]): #if array reversed
                   flux    = obs_flux[ind][::-1]
                   fluxerr = obs_fluxerr[ind][::-1]
                   DQ      = old_DQ[ind][::-1]
                else:
                   flux    = obs_flux[ind]
                   fluxerr = obs_fluxerr[ind]
                   DQ      = obs_DQ[ind]
            else:
                for l in range(len(wavelength)):
                    if (wave[ind][nzero][0] > wave[ind][nzero][-1]):
                        #TVM: Changed reversed indices from [:,-l] to [:,-l-1] so it starts on [-1] not [-0]
                        flux[l]    = np.mean(obs_flux[:,-l-1]) #take mean of each flux array if there are multiple
                        fluxerr[l] = np.sqrt(sum(obs_fluxerr[:,-l-1]**2)) #/ (obs_fluxerr.shape[0]-1) #sqrt of variance / n-1
                        DQ[l]      = sum(obs_DQ[:,-l-1])
                    else:
                        flux[l]    = np.mean(obs_flux[:,l])
                        fluxerr[l] = np.sqrt(sum(obs_fluxerr[:,l]**2)) #/ (obs_fluxerr.shape[0]-1)
                        DQ[l]      = sum(obs_DQ[:,l])
        else:
            if (wave[nzero][0] > wave[nzero][-1]):
                flux    = obs_flux[::-1]
                fluxerr = obs_fluxerr[::-1]
                DQ      = obs_DQ[::-1]
            else:
                flux    = obs_flux
                fluxerr = obs_fluxerr
                DQ      = obs_DQ


        '''
        Step 2: Masks bad pixels -- both as classified by HST and those
                in overlapping skyline regions.

                Also cut edge pixels with consecutive
                1) Negative waves (why does this happen??)
                2) Negative fluxes
                3) Masked pixels
                Note that this is relevant as the edges of the spectra
                will be lower S/N.
        '''
        #Mask bad pixels
        sel                       = (wavelength >= 1205) & (wavelength <= 1225) #geocoronal lines
        mask                      = DQ>0 #changed by TVM since individual pixels are being summed #(DQ==800) | (DQ==700) | (DQ==400) | (DQ==300) | (DQ==200)
        flux_wmask                = flux.copy()
        #flux_wmask[mask|sel]      = np.nan
        err_wmask                 = fluxerr.copy()
        #err_wmask[mask|sel]       = 0.
        masks[i,:len(err_wmask)][err_wmask==0.] = 1

        #Finish off pre-processing
        waves[i,:len(flux)]     = Cut_Edge_Pix_TVM.Cut_Edge_Pix(DQ, wavelength, flux_wmask, err_wmask, \
                                                wavelength, array_len, "wavelength", False, z, "%s - %s"%(Identifier,spec_names[i]),"FOS")
        fluxes[i,:len(flux)]    = Cut_Edge_Pix_TVM.Cut_Edge_Pix(DQ, wavelength, flux_wmask, err_wmask, \
                                                flux_wmask, array_len, "flux", False, z, "%s - %s"%(Identifier,spec_names[i]),"FOS")
        flux_errs[i,:len(flux)] = Cut_Edge_Pix_TVM.Cut_Edge_Pix(DQ, wavelength, flux_wmask, err_wmask, \
                                                err_wmask, array_len, "flux error", False, z, "%s - %s"%(Identifier,spec_names[i]),"FOS")
        masks[i,:len(flux)]     = Cut_Edge_Pix_TVM.Cut_Edge_Pix(DQ, wavelength, flux_wmask, err_wmask, \
                                                masks[i,:len(err_wmask)], array_len, "masks", False, z, "%s - %s"%(Identifier,spec_names[i]),"FOS")

    return waves, fluxes, flux_errs, masks
