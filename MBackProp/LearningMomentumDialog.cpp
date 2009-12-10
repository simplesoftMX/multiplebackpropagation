/*
	Noel Lopes is a Professor Assistant at the Polytechnic Institute of Guarda, Portugal (for more information see readme.txt)
    Copyright (C) 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009 Noel de Jesus Mendonça Lopes

	This file is part of Multiple Back-Propagation.

    Multiple Back-Propagation is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

/**
 Class    : LearningMomentumDialog
 Puropse  : Learning rate and momentum adjustment dialog class.
 Date     : 3 of April of 2000
 Reviewed : 14 of June of 2000
 Version  : 1.0.0
 Comments :
             ---------
            | CObject |
             ---------
                |   ------------
                -->| CCmdTarget |
                    ------------
                      |   ------
                      -->| CWnd |
                          ------
                            |   ---------
                            -->| CDialog |
                                ---------
                                  |   ------------------------
                                  -->| LearningMomentumDialog |
                                      ------------------------
*/
#include "stdafx.h"
#include "MBackProp.h"
#include "LearningMomentumDialog.h"

#include "cuda.h"

#ifdef _DEBUG
	#define new DEBUG_NEW
	#undef THIS_FILE
	static char THIS_FILE[] = __FILE__;
#endif

BEGIN_MESSAGE_MAP(LearningMomentumDialog, CDialog)
	//{{AFX_MSG_MAP(LearningMomentumDialog)
	ON_BN_CLICKED(IDC_BATCH, OnBatch)
	ON_BN_CLICKED(IDC_ONLINE, OnOnline)
	ON_BN_CLICKED(IDC_AUTO_UPDATE_LEARNING, OnAutoUpdateLearning)
	ON_BN_CLICKED(IDC_AUTO_UPDATE_MOMENTUM, OnAutoUpdateMomentum)
	ON_BN_CLICKED(IDC_DELTA_BAR_DELTA, OnDeltaBarDeltaChange)
	ON_BN_CLICKED(IDC_ROBUSTNESS, OnRobustness)
	ON_BN_CLICKED(IDC_STOP_RMS, OnStopRms)
	ON_BN_CLICKED(IDC_STOP_EPOCHS, OnStopEpochs)
	//}}AFX_MSG_MAP
	ON_BN_CLICKED(IDC_ENABLE_CUDA, &LearningMomentumDialog::OnBnClickedEnableCuda)
END_MESSAGE_MAP()

/**
 Method  : void LearningMomentumDialog::OnOK()
 Purpose : Change the learning rate and the momentum.
 Version : 1.0.0
*/
void LearningMomentumDialog::OnOK() {
	parent->mainNetLearningMomentumInformation.learningRate.value = mainLearning.GetValue();
	parent->mainNetLearningMomentumInformation.learningRate.decayPercentage = mainLearningPercentageDecay.GetValue();
	parent->mainNetLearningMomentumInformation.learningRate.decayEpochs = mainLearningEpochDecay.GetValue();

	parent->mainNetLearningMomentumInformation.momentum.value = mainMomentum.GetValue();
	parent->mainNetLearningMomentumInformation.momentum.decayPercentage = mainMomentumPercentageDecay.GetValue();
	parent->mainNetLearningMomentumInformation.momentum.decayEpochs = mainMomentumEpochDecay.GetValue();

	parent->spaceNetLearningMomentumInformation.learningRate.value = spaceLearning.GetValue();
	parent->spaceNetLearningMomentumInformation.learningRate.decayPercentage = spaceLearningPercentageDecay.GetValue();
	parent->spaceNetLearningMomentumInformation.learningRate.decayEpochs = spaceLearningEpochDecay.GetValue();

	parent->spaceNetLearningMomentumInformation.momentum.value = spaceMomentum.GetValue();
	parent->spaceNetLearningMomentumInformation.momentum.decayPercentage = spaceMomentumPercentageDecay.GetValue();
	parent->spaceNetLearningMomentumInformation.momentum.decayEpochs = spaceMomentumEpochDecay.GetValue();

	parent->epochsStop = (GetCheckedRadioButton(IDC_STOP_RMS, IDC_STOP_EPOCHS) == IDC_STOP_EPOCHS);

	parent->numberEpochsToStop = epochsToStop.GetValue();

	parent->rmsStop = rmsToStop.GetValue();

	parent->spaceRmsStop = spaceRmsToStop.GetValue();

	parent->batchTraining = (GetCheckedRadioButton(IDC_BATCH, IDC_ONLINE) == IDC_BATCH);

	parent->deltaBarDelta = (deltaBarDeltaButton.GetCheck() > 0);

	parent->randomizePatterns = (RandomizePatterns.GetCheck() > 0);

	parent->autoUpdateLearning = (!parent->deltaBarDelta && (autoUpdateLearningButton.GetCheck() > 0));

	parent->autoUpdateMomentum = (autoUpdateMomentumButton.GetCheck() > 0);

	parent->UpdateLearningMomentum();

	Connection::u = u.GetValue();
	Connection::d = d.GetValue();
	Connection::maxStepSize = maxStepSize.GetValue();

	parent->robustLearning = (useRobustness.GetCheck() > 0);

	parent->rmsGrowToApplyRobustLearning = 1.0 + (applyRobustnessWhenRMSGrows.GetValue() / 100);

	parent->robustFactor = robustnessReduceFactor.GetValue();

	parent->weightDecay = weightDecay.GetValue();

	switch (comboThread.GetCurSel()) {
		case 0:
			parent->programPriority = NORMAL_PRIORITY_CLASS;
			break;
		case 1:
			parent->programPriority = HIGH_PRIORITY_CLASS;
			break;
		case 2:
			parent->programPriority = REALTIME_PRIORITY_CLASS;
	}
	
	parent->updateScreen = (updateScreen.GetCheck() > 0);

	#ifdef MBP_WITH_CUDA
		parent->useCuda = (cbCUDA.GetCheck() > 0);
		parent->MBPTopologyCtrl->SetCudaRestrictions((parent->useCuda) ? 1 : 0);
	#endif

	CDialog::OnOK();
}

/**
 Method  : void LearningMomentumDialog::DoDataExchange(CDataExchange* pDX)
 Purpose : Exchange data.
 Version : 1.0.0
*/
void LearningMomentumDialog::DoDataExchange(CDataExchange* pDX) {
	CDialog::DoDataExchange(pDX);

	DDX_Control(pDX, IDC_THREAD, comboThread);
	DDX_Control(pDX, IDC_UPDATE_SCREEN, updateScreen);
	DDX_Control(pDX, IDC_LB_SM, lbSpaceMomentum);
	DDX_Control(pDX, IDC_LB_SLR, lbSpaceLearning);
	DDX_Control(pDX, IDC_LB_MM, lbMainMomentum);
	DDX_Control(pDX, IDC_LB_MLR, lbMainLearning);
	DDX_Control(pDX, IDC_ROBUSTNESS, useRobustness);
	DDX_Control(pDX, IDC_DELTA_BAR_DELTA, deltaBarDeltaButton);
	DDX_Control(pDX, IDC_AUTO_UPDATE_MOMENTUM, autoUpdateMomentumButton);
	DDX_Control(pDX, IDC_AUTO_UPDATE_LEARNING, autoUpdateLearningButton);
	DDX_Control(pDX, IDC_RANDOMIZE_PATTERNS, RandomizePatterns);
	DDX_Control(pDX, IDC_MAIN_NET_LEARN_EPOCH_DECAY, mainLearningEpochDecay);
	DDX_Control(pDX, IDC_MAIN_NET_LEARN_PERC_DECAY, mainLearningPercentageDecay);
	DDX_Control(pDX, IDC_MAIN_NET_LEARNING, mainLearning);
	DDX_Control(pDX, IDC_MAIN_NET_MOM_EPOCH_DECAY, mainMomentumEpochDecay);
	DDX_Control(pDX, IDC_MAIN_NET_MOM_PERC_DECAY, mainMomentumPercentageDecay);
	DDX_Control(pDX, IDC_MAIN_NET_MOMENTUM, mainMomentum);
	DDX_Control(pDX, IDC_SPACE_NET_LEARNIG, spaceLearning);
	DDX_Control(pDX, IDC_SPACE_NET_LEARNIG_EPOCH_DECAY, spaceLearningEpochDecay);
	DDX_Control(pDX, IDC_SPACE_NET_LEARNIG_PERC_DECAY, spaceLearningPercentageDecay);
	DDX_Control(pDX, IDC_SPACE_NET_MOMENTUM, spaceMomentum);
	DDX_Control(pDX, IDC_SPACE_NET_MOMENTUM_EPOCH_DECAY, spaceMomentumEpochDecay);
	DDX_Control(pDX, IDC_SPACE_NET_MOMENTUM_PERC_DECAY, spaceMomentumPercentageDecay);
	DDX_Control(pDX, IDC_EPOCHS_TO_STOP, epochsToStop);
	DDX_Control(pDX, IDC_RMS_TO_STOP, rmsToStop);
	DDX_Control(pDX, IDC_DOWN_DBD, d);
	DDX_Control(pDX, IDC_UP_DBD, u);
	DDX_Control(pDX, IDC_MAX_STEPSIZE, maxStepSize);
	DDX_Control(pDX, IDC_ROBUSTNESS_LR_REDUCE, robustnessReduceFactor);
	DDX_Control(pDX, IDC_RMS_GROW_PERCENT, applyRobustnessWhenRMSGrows);
	DDX_Control(pDX, IDC_WEIGHT_DECAY, weightDecay);
	DDX_Control(pDX, IDC_SPACE_RMS_TO_STOP, spaceRmsToStop);
	DDX_Control(pDX, IDC_ENABLE_CUDA, cbCUDA);
	DDX_Control(pDX, IDC_BATCH, radioBatch);
	DDX_Control(pDX, IDC_ONLINE, radioOnline);
}

#ifdef MBP_WITH_CUDA

void LearningMomentumDialog::CheckForInvalidLayersCuda(bool network) {
	int layers = parent->MBPTopologyCtrl->GetLayers(network);

	for (int l = 0; l < layers; l++) {
		int neurons = parent->MBPTopologyCtrl->GetNeurons(l, network);

		for (int n = 0; n < neurons; n++) {
			if (parent->MBPTopologyCtrl->GetActivationFunction(l, n, network) != (long) Sigmoid) {
				invalidActivationFunctionsCuda = true;
				return;
			}

			if (parent->MBPTopologyCtrl->GetActivationFunctionParameter(l, n, network) != 1.0) {
				invalidActivationFunctionsCuda = true;
				return;
			}
		}
	}
}

#endif


/**
 Method  : void LearningMomentumDialog::DoDataExchange(CDataExchange* pDX)
 Purpose : Initialize the dialog.
 Version : 1.0.0
*/
BOOL LearningMomentumDialog::OnInitDialog() {
	CDialog::OnInitDialog();

	#ifdef MBP_WITH_CUDA

	hasConnectionsBetweenInputsAndOutputs = false;
	invalidNetworkTypeCuda = false;
	invalidActivationFunctionsCuda = false;

	makeChangesIfNeededForCuda = false;

	if (parent->MBPTopologyCtrl->GetConnectInputLayerWithOutputLayer(TRUE) || parent->MBPTopologyCtrl->GetConnectInputLayerWithOutputLayer(FALSE)) hasConnectionsBetweenInputsAndOutputs = true;
	if (parent->MBPTopologyCtrl->GetNetworkType() == MBPHO) invalidNetworkTypeCuda = true;

	CheckForInvalidLayersCuda(true);
	if (!invalidActivationFunctionsCuda) CheckForInvalidLayersCuda(false);

	#endif

	//mainLearning.SetValue(parent->mainNetLearningMomentumInformation.learningRate.value);
	mainLearning.SetValue(parent->editMainLearning.GetValue());
	mainLearningPercentageDecay.SetValue(parent->mainNetLearningMomentumInformation.learningRate.decayPercentage);
	mainLearningEpochDecay.SetValue(parent->mainNetLearningMomentumInformation.learningRate.decayEpochs);

	//mainMomentum.SetValue(parent->mainNetLearningMomentumInformation.momentum.value);
	mainMomentum.SetValue(parent->editMainMomentum.GetValue());
	mainMomentumPercentageDecay.SetValue(parent->mainNetLearningMomentumInformation.momentum.decayPercentage);
	mainMomentumEpochDecay.SetValue(parent->mainNetLearningMomentumInformation.momentum.decayEpochs);

	//spaceLearning.SetValue(parent->spaceNetLearningMomentumInformation.learningRate.value);
	spaceLearning.SetValue(parent->editSpaceLearning.GetValue());
	spaceLearningPercentageDecay.SetValue(parent->spaceNetLearningMomentumInformation.learningRate.decayPercentage);
	spaceLearningEpochDecay.SetValue(parent->spaceNetLearningMomentumInformation.learningRate.decayEpochs);

	//spaceMomentum.SetValue(parent->spaceNetLearningMomentumInformation.momentum.value);
	spaceMomentum.SetValue(parent->editSpaceMomentum.GetValue());
	spaceMomentumPercentageDecay.SetValue(parent->spaceNetLearningMomentumInformation.momentum.decayPercentage);
	spaceMomentumEpochDecay.SetValue(parent->spaceNetLearningMomentumInformation.momentum.decayEpochs);

	epochsToStop.SetValue(parent->numberEpochsToStop);

	rmsToStop.SetValue(parent->rmsStop);

	spaceRmsToStop.SetValue(parent->spaceRmsStop);

	CheckRadioButton(IDC_STOP_RMS, IDC_STOP_EPOCHS, (parent->epochsStop) ? IDC_STOP_EPOCHS : IDC_STOP_RMS);

	if (parent->batchTraining) {
		CheckRadioButton(IDC_BATCH, IDC_ONLINE, IDC_BATCH);
		RandomizePatterns.EnableWindow(FALSE);
	} else {
		CheckRadioButton(IDC_BATCH, IDC_ONLINE, IDC_ONLINE);
		RandomizePatterns.SetCheck((parent->randomizePatterns) ? 1 : 0);
	}

	autoUpdateLearningButton.SetCheck((parent->autoUpdateLearning) ? 1 : 0);
	OnAutoUpdateLearning();
	autoUpdateMomentumButton.SetCheck((parent->autoUpdateMomentum) ? 1 : 0);	
	OnAutoUpdateMomentum();
	deltaBarDeltaButton.SetCheck((parent->deltaBarDelta) ? 1 : 0); 
	OnDeltaBarDeltaChange();

	u.SetValue(Connection::u);
	d.SetValue(Connection::d);
	maxStepSize.SetValue(Connection::maxStepSize);

	useRobustness.SetCheck((parent->robustLearning) ? 1 : 0); 
	OnRobustness();

	applyRobustnessWhenRMSGrows.SetValue((parent->rmsGrowToApplyRobustLearning - 1.0) * 100);
	robustnessReduceFactor.SetValue(parent->robustFactor);

	weightDecay.SetValue(parent->weightDecay);

	if (parent->epoch > 0) {
		lbMainLearning.SetWindowText(_TEXT("Current learning rate"));
		lbMainMomentum.SetWindowText(_TEXT("Current momentum"));
		lbSpaceLearning.SetWindowText(_TEXT("Current learning rate"));
		lbSpaceMomentum.SetWindowText(_TEXT("Current momentum"));
	}

	switch (parent->programPriority) {
		case NORMAL_PRIORITY_CLASS :
			comboThread.SetCurSel(0);
			break;
		case HIGH_PRIORITY_CLASS :
			comboThread.SetCurSel(1);
			break;
		case REALTIME_PRIORITY_CLASS:
			comboThread.SetCurSel(2);
	}

	updateScreen.SetCheck((parent->updateScreen) ? 1 : 0);

	#ifdef MBP_WITH_CUDA
		cbCUDA.EnableWindow(parent->cuda.Supported());
		cbCUDA.SetCheck((parent->useCuda) ? 1: 0);
		OnBnClickedEnableCuda();
	#endif

	return TRUE;
}

BEGIN_EVENTSINK_MAP(LearningMomentumDialog, CDialog)
    //{{AFX_EVENTSINK_MAP(LearningMomentumDialog)
	ON_EVENT(LearningMomentumDialog, IDC_RMS_TO_STOP, 1 /* Change */, OnChangeRMSToStop, VTS_NONE)
	ON_EVENT(LearningMomentumDialog, IDC_EPOCHS_TO_STOP, 1 /* Change */, OnChangeEpochsToStop, VTS_NONE)
	//}}AFX_EVENTSINK_MAP
END_EVENTSINK_MAP()

void LearningMomentumDialog::OnChangeRMSToStop() {
	CheckRadioButton(IDC_STOP_RMS, IDC_STOP_EPOCHS, IDC_STOP_RMS);
}

void LearningMomentumDialog::OnChangeEpochsToStop() {
	CheckRadioButton(IDC_STOP_RMS, IDC_STOP_EPOCHS, IDC_STOP_EPOCHS);
}

void LearningMomentumDialog::OnBatch() {
	RandomizePatterns.SetCheck(0);
	RandomizePatterns.EnableWindow(FALSE);
}

void LearningMomentumDialog::OnOnline() {
	RandomizePatterns.EnableWindow(TRUE);
}

void LearningMomentumDialog::OnAutoUpdateLearning() {
	BOOL autoUpdate = (autoUpdateLearningButton.GetCheck() > 0);

	mainLearningPercentageDecay.EnableWindow(!autoUpdate);
	mainLearningEpochDecay.EnableWindow(!autoUpdate);

	spaceLearningPercentageDecay.EnableWindow(!autoUpdate);
	spaceLearningEpochDecay.EnableWindow(!autoUpdate);
}

void LearningMomentumDialog::OnAutoUpdateMomentum() {
	BOOL autoUpdate = (autoUpdateMomentumButton.GetCheck() > 0);

	mainMomentumPercentageDecay.EnableWindow(!autoUpdate);
	mainMomentumEpochDecay.EnableWindow(!autoUpdate);

	spaceMomentumPercentageDecay.EnableWindow(!autoUpdate);
	spaceMomentumEpochDecay.EnableWindow(!autoUpdate);	
}

void LearningMomentumDialog::OnDeltaBarDeltaChange() {
	BOOL adaptiveStepSizes = (deltaBarDeltaButton.GetCheck() > 0);	

	if (adaptiveStepSizes) {
		autoUpdateLearningButton.SetCheck(1);
		OnAutoUpdateLearning();
		autoUpdateLearningButton.EnableWindow(FALSE);
		u.EnableWindow(TRUE);
		d.EnableWindow(TRUE);
		mainLearning.EnableWindow(parent->epoch == 0);
		spaceLearning.EnableWindow(parent->epoch == 0);
	} else {
		autoUpdateLearningButton.EnableWindow(TRUE);
		u.EnableWindow(FALSE);
		d.EnableWindow(FALSE);
		mainLearning.EnableWindow(TRUE);
		spaceLearning.EnableWindow(TRUE);
	}
}

void LearningMomentumDialog::OnRobustness() {
	BOOL robustness = (useRobustness.GetCheck() > 0);	

	applyRobustnessWhenRMSGrows.EnableWindow(robustness);
	robustnessReduceFactor.EnableWindow(robustness);	
}

void LearningMomentumDialog::OnStopRms() {
	CheckRadioButton(IDC_STOP_RMS, IDC_STOP_EPOCHS, IDC_STOP_RMS);
}

void LearningMomentumDialog::OnStopEpochs() {
	CheckRadioButton(IDC_STOP_RMS, IDC_STOP_EPOCHS, IDC_STOP_EPOCHS);
}

void LearningMomentumDialog::OnBnClickedEnableCuda() {
	#ifdef MBP_WITH_CUDA

	BOOL cudaEnabled = (cbCUDA.GetCheck() > 0);

	if (cudaEnabled) {
		if (!makeChangesIfNeededForCuda) {
			if (hasConnectionsBetweenInputsAndOutputs || invalidNetworkTypeCuda || invalidActivationFunctionsCuda) {
				CString info = L"Currently, the following limitations apply when using CUDA:\n\n";

				if (hasConnectionsBetweenInputsAndOutputs) info += L"- Connections between the input and the output layers are not supported.\n";
				if (invalidNetworkTypeCuda) info += L"- The type of network selected is not supported.\n";
				if (invalidActivationFunctionsCuda) info += L"- The only activation function supported is the logistic (sigmoid) function with a parameter of 1.0.\n";

				info += L"\nDo you want MBP to change your network topology so you can use CUDA to train it?";

				if (MessageBox(info, L"MBP CUDA limitations", MB_YESNO | MB_ICONEXCLAMATION) == IDYES) {
					makeChangesIfNeededForCuda = true;
				}else {
					cbCUDA.SetCheck(BST_CHECKED);
					return;
				}
			}
		}

		deltaBarDeltaButton.SetCheck(TRUE);
		OnDeltaBarDeltaChange();

		autoUpdateMomentumButton.SetCheck(FALSE);
		OnAutoUpdateMomentum();

		mainMomentumPercentageDecay.SetValue(0);
		spaceMomentumPercentageDecay.SetValue(0);

		spaceRmsToStop.SetValue(0.0);

		CheckRadioButton(IDC_BATCH, IDC_ONLINE, IDC_BATCH);
		OnBatch();

		weightDecay.SetValue(0.0);

		updateScreen.SetCheck(FALSE);
	}

	deltaBarDeltaButton.EnableWindow(!cudaEnabled);
	autoUpdateMomentumButton.EnableWindow(!cudaEnabled);

	if (autoUpdateMomentumButton.GetCheck() == 0) {
		mainMomentumEpochDecay.EnableWindow(!cudaEnabled);
		mainMomentumPercentageDecay.EnableWindow(!cudaEnabled);
		spaceMomentumEpochDecay.EnableWindow(!cudaEnabled);
		spaceMomentumPercentageDecay.EnableWindow(!cudaEnabled);
	}

	spaceRmsToStop.EnableWindow(!cudaEnabled);
	radioBatch.EnableWindow(!cudaEnabled);	
	radioOnline.EnableWindow(!cudaEnabled);
	updateScreen.EnableWindow(!cudaEnabled);
	weightDecay.EnableWindow(!cudaEnabled);

	#endif
}