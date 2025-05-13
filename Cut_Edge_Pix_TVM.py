#Cut_Edge_Pix.py

import numpy as np
#import SpecCuts #specific cuts on individual exposures

#Need an additional clause to see if the

def maskEdge(wave, flux, errs, mask):
    '''
    mask the edges of (mainly SDSS) spectrum so we ignore it in the ICA fitting
    '''
    s2n = flux / errs
    good_mask = ( (flux!=0) & (~np.isnan(flux)) & (wavelength>0) & (fluxerr!=0) )
    istart = (np.abs(wavelength - wavelength[good_mask][0])).argmin()
    iend   = (np.abs(wavelength - wavelength[good_mask][-1])).argmin()
    #while s2n[iend-30:iend+30]


def Cut_Edge_Pix(DQ,wavelength,flux,fluxerr,input_data,array_len,data_type,Cut_Spikes,z=0,exp_id=None,Instrument=None):
    #print(Instrument)
    if Instrument=="FOS":
        import SpecCuts_FOS as SpecCuts
    elif Instrument=="STIS":
        import SpecCuts_STIS as SpecCuts
    elif Instrument=="COS":
        import SpecCuts_COS as SpecCuts
    else:
        import SpecCuts_HSLA as SpecCuts

    #Also cut if SNR is bad? Might fix my bad exposure problem...
    #The issue now is that I'm removing pixels based on different criteria instead of any...change?

    #13-24 just set negative wavelengths to 0:
    all_snrs = np.zeros(len(flux))
    for i in range(len(fluxerr)):
        all_snrs[i] = np.nan if fluxerr[i] == 0 else flux[i]/fluxerr[i]
    if exp_id in SpecCuts.BlueEdges and exp_id in SpecCuts.RedEdges:
        wavecut_blue = SpecCuts.BlueEdges[exp_id] * (1+z)
        wavecut_red  = SpecCuts.RedEdges[exp_id] * (1+z)
        good_mask = ( (flux!=0) & (~np.isnan(flux)) & (wavelength>0) & (fluxerr!=0) &
                        (all_snrs>1.5) & (wavelength>wavecut_blue) & (wavelength<wavecut_red) )
    elif exp_id in SpecCuts.BlueEdges:
        wavecut_blue = SpecCuts.BlueEdges[exp_id] * (1+z)
        good_mask = ( (flux!=0) & (~np.isnan(flux)) & (wavelength>0) & (fluxerr!=0) & (all_snrs>1.5) & (wavelength>wavecut_blue) )
    elif exp_id in SpecCuts.RedEdges:
        wavecut_red = SpecCuts.RedEdges[exp_id] * (1+z)
        good_mask = ( (flux!=0) & (~np.isnan(flux)) & (wavelength>0) & (fluxerr!=0) & (all_snrs>1.5) & (wavelength<wavecut_red) )
    else:
        good_mask = ( (flux!=0) & (~np.isnan(flux)) & (wavelength>0) & (fluxerr!=0) & (all_snrs>1.5) )

    #good_mask = ( (flux!=0) & (~np.isnan(flux)) & (wavelength>0) & (fluxerr!=0) & (all_snrs>2.5) ) #& (DQ==0)
    #print(wavelength, good_mask.sum())
    begin_stop_index = (np.abs(wavelength - wavelength[good_mask][0])).argmin()
    end_stop_index = (np.abs(wavelength - wavelength[good_mask][-1])).argmin()
    #FIXME: maybe change so that most pixels have S2N > 2 or something?
    while (all_snrs[begin_stop_index:begin_stop_index+65]<=0).sum()>20:
        begin_stop_index += 1
    while (all_snrs[end_stop_index-65:end_stop_index]<=0).sum()>20:
        end_stop_index -= 1
    """
    for i in range(len(wavelength)-1):
        if (wavelength[i] < 0 and wavelength[i+1] < 0):
            continue
        elif (flux[i+1]==0 or np.isnan(flux[i+1])):
            continue
        elif wavelength[i]<0 and wavelength[i+1] >= 0:
            begin_stop_index = i+1
            break
        else:
            begin_stop_index = (np.abs(wavelength - wavelength[( (flux!=0) )][0])).argmin()
            #begin_stop_index = 0
    """

    if data_type == "wavelength":
        trunc_data = np.array(list(np.zeros(begin_stop_index)) + list(input_data[begin_stop_index:]))
        return trunc_data

    if ~Cut_Spikes:
        #Save SNRs for each pixel
        all_snrs = np.zeros(len(flux))
        for i in range(len(fluxerr)):
            all_snrs[i] = np.nan if fluxerr[i] == 0 else flux[i]/fluxerr[i]

        #Determine where first good pixel is
        """
        end_stop_index = None
        for i in range(begin_stop_index, len(all_snrs)-1):
            if DQ[i] > 0 and DQ[i+1] > 0:
                continue
            elif DQ[i]>0 and DQ[i+1]==0:
                begin_stop_index = i+1
                break
            else:
                begin_stop_index = 0

        #Determine where last good pixel is
        for i in range(len(all_snrs)-1):
            if DQ[-i-1] > 0 and DQ[-i-2] > 0:
                continue
            elif DQ[-i-1]>0 and DQ[-i-2]==0:
                end_stop_index = -i-2
                break
            else:
                end_stop_index = None
        """

        #print("Cut Edge Pix test")
        #print(begin_stop_index, end_stop_index)
        #print(begin_stop_index_tvm, end_stop_index_tvm)

        if data_type == "flux":
            trunc_data = np.concatenate((np.nan*np.ones(begin_stop_index), input_data[begin_stop_index:]))
            trunc_data[end_stop_index:] = np.nan if end_stop_index is not None else trunc_data[end_stop_index:]

        elif data_type == "flux error":
            trunc_data = np.concatenate((np.zeros(begin_stop_index), input_data[begin_stop_index:]))
            trunc_data[end_stop_index:] = 0. if end_stop_index is not None else trunc_data[end_stop_index:]

        if data_type == "masks":
            trunc_data = np.concatenate((np.zeros(begin_stop_index), input_data[begin_stop_index:]))
            trunc_data[end_stop_index:] = 0. if end_stop_index is not None else trunc_data[end_stop_index:]

        return trunc_data
