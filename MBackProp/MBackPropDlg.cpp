/*
	Noel Lopes is an Assistant Professor at the Polytechnic Institute of Guarda, Portugal (for more information see readme.txt)
	Copyright (C) 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011 Noel de Jesus Mendonça Lopes

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
 Class    : CMBackPropDlg
 Purpose  : Represents the Back-Propagation dialog.
 Date     : 3 of July of 1999.
 Reviewed : 14 of March of 2009.
 Version  : 1.8.0
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
								  |   --------------
								  -->| CMBackPropDlg |
									  --------------
*/
#include "stdafx.h"
#include "MBackProp.h"
#include "MBackPropDlg.h"
#include "BackPropagation.h"
#include "RandWeightsDialog.h"
#include "SaveWeightsDialog.h"
#include "LoadWeightsDialog.h"
#include "SaveNetworkDialog.h"
#include "LoadNetworkDialog.h"
#include "LearningMomentumDialog.h"
#include "WeightsDialog.h"
#include "SensitivityDialog.h"
#include "GenerateCCodeDialog.h"
#include "AditionalDefinitions.h"
#include "../Common/Files/OutputFile.h"

#include <io.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <assert.h>
#include <errno.h>
#include <process.h>
#include <time.h>
#include <afxadv.h>
#include <afxole.h>

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

#ifdef NOEL_USE_ONLY
	void CopyStringToClipboard(CString s) {
		CSharedFile	mem(GMEM_MOVEABLE | GMEM_SHARE);
		mem.Write(s, (s.GetLength() + 1) * sizeof(TCHAR));
		HGLOBAL handleMem = mem.Detach();

		if (handleMem) {
			COleDataSource * dataSource = new COleDataSource;
			dataSource->CacheGlobalData(CF_UNICODETEXT, handleMem);
			dataSource->SetClipboard();
		}
	}
#endif

/**
 Attribute : static char * unrecognizedFormat
 Purpose   : Contains a string specifying that 
			   the file format is not recognized.
 Comments  : static attributte.
*/
const char * CMBackPropDlg::unrecognizedFormat = "Can not recognize the file format.";

/**
 Attribute : static const int maxPointsRMS
 Purpose   : Contains the maximum number of 
			 points used to draw the graphics.
 Comments  : static attributte.
*/
const int CMBackPropDlg::maxPointsRMS = 512;

/**
 Constructor : CMBackPropDlg::CMBackPropDlg()
 Purpose     : Create a Multiple Back-Propagation dialog.
 Version     : 1.1.0
*/
CMBackPropDlg::CMBackPropDlg() : CDialog(CMBackPropDlg::IDD) {
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);

	#ifdef MBP_WITH_CUDA		
		useCuda = cuda.Supported();
	#else
		bool useCuda = false;
	#endif

	mainNetLearningMomentumInformation.learningRate.value = 0.7;
	mainNetLearningMomentumInformation.learningRate.decayEpochs = 1000;
	mainNetLearningMomentumInformation.learningRate.decayPercentage = 1;

	mainNetLearningMomentumInformation.momentum.value = 0.7;
	mainNetLearningMomentumInformation.momentum.decayEpochs = 1000;
	mainNetLearningMomentumInformation.momentum.decayPercentage = (useCuda) ? 0 : 1;

	spaceNetLearningMomentumInformation.learningRate.value = 0.7;
	spaceNetLearningMomentumInformation.learningRate.decayEpochs = 1000;
	spaceNetLearningMomentumInformation.learningRate.decayPercentage = 1;

	spaceNetLearningMomentumInformation.momentum.value = 0.7;
	spaceNetLearningMomentumInformation.momentum.decayEpochs = 1000;
	spaceNetLearningMomentumInformation.momentum.decayPercentage = (useCuda) ? 0 : 1;

	updateScreen = !useCuda;
	programPriority = NORMAL_PRIORITY_CLASS;
	epoch = 0;
	trainingTime = 0;
	reloadTrainingData = true;
	reloadTestingData  = false;
	autoUpdateLearning = true;
	autoUpdateMomentum = false; // changed on version 2.0.6
	epochsStop = false;
	numberEpochsToStop = 1000000;
	rmsStop = 0.01;
	spaceRmsStop = 0.0;
	randomizePatterns = !useCuda; // changed on version 2.0.6
	batchTraining = useCuda; // changed on version 2.0.6
	robustLearning = true;
	rmsGrowToApplyRobustLearning = 1.001;
	robustFactor = 0.5;
	deltaBarDelta = true;
	weightDecay = 0.0;

	percentIncDecLearnRate = percentIncDecSpaceLearnRate = 0.01;
	percentIncDecMomentum = percentIncDecSpaceMomentum = 0.01;

	trainingThread = NULL;
}

/**
 Method  : void CMBackPropDlg::DoDataExchange(CDataExchange* pDX)
 Purpose : Exchange data.
 Version : 1.0.0
*/
void CMBackPropDlg::DoDataExchange(CDataExchange* pDX) {
	CDialog::DoDataExchange(pDX);

	DDX_Control(pDX, IDSENSIVITY, sensitivityButton);
	DDX_Control(pDX, IDC_LABEL_TRAINRMSDISPLAY_SPACE, labelSpaceTrainRMSDisplay);
	DDX_Control(pDX, IDC_LABEL_TESTRMSDISPLAY_SPACE, labelSpaceTestRMSDisplay);
	DDX_Control(pDX, IDC_TEST_RMS_SPACE, testSpaceRMSDisplay);
	DDX_Control(pDX, IDC_TRAIN_RMS_SPACE, trainSpaceRMSDisplay);
	DDX_Control(pDX, IDC_MAINNETRMSFRAME, frameMainRMS);
	DDX_Control(pDX, IDC_SPACENETRMSFRAME, frameSpaceRMS);
	DDX_Control(pDX, IDVIEWWEIGHTS, viewWeights);
	DDX_Control(pDX, IDC_LABEL_SPACE_MOMENTUM, labelSpaceMomentum);
	DDX_Control(pDX, IDLEARNINGMOMENTUM, LearningMomentum);
	DDX_Control(pDX, IDC_SPACENETLEARNINGFRAME, frameSpaceLearning);
	DDX_Control(pDX, IDC_MAINNETLEARNINGFRAME, frameMainLearning);
	DDX_Control(pDX, IDC_LABEL_SPACE_LEARNING, labelSpaceLearning);
	DDX_Control(pDX, IDC_LABEL_MAIN_MOMENTUM, labelMainMomentum);
	DDX_Control(pDX, IDC_LABEL_MAIN_LEARNING, labelMainLearning);
	DDX_Control(pDX, IDC_FRAMELEARNING, frameLearning);
	DDX_Control(pDX, IDSAVENETWORK, saveNetwork);
	DDX_Control(pDX, IDLOADNETWORK, loadNetwork);
	DDX_Control(pDX, IDCCODE, generateCCode);
	DDX_Control(pDX, IDC_FRAME_NETWORK, frameNetwork);
	DDX_Control(pDX, IDSAVEWEIGHTS, saveWeightsButton);
	DDX_Control(pDX, IDRANDOMIZEWEIGHTS, randomizeWeightsButton);
	DDX_Control(pDX, IDLOADWEIGHTS, loadWeightsButton);
	DDX_Control(pDX, IDC_FRAME_WEIGHTS, frameWeights);
	DDX_Control(pDX, IDC_LABEL_TRAINRMSDISPLAY, labelTrainRMSDisplay);
	DDX_Control(pDX, IDC_LABEL_TESTRMSDISPLAY, labelTestRMSDisplay);
	DDX_Control(pDX, IDC_FRAMEEPOCH, frameEpoch);
	DDX_Control(pDX, IDC_FRAMERMS, frameRMS);
	DDX_Control(pDX, IDC_FRAMEDATAFILES, frameDataFiles);
	DDX_Control(pDX, IDC_TAB, tabs);
	DDX_Control(pDX, IDSTOP, stopButton);
	DDX_Control(pDX, IDC_TRAIN_RMS, trainRMSDisplay);
	DDX_Control(pDX, IDC_TEST_RMS, testRMSDisplay);
	DDX_Control(pDX, IDC_EPOCH, epochDisplay);
	DDX_Control(pDX, IDTRAIN, trainButton);
	DDX_Control(pDX, IDC_TrainFileBox, m_trainFileBox);
	DDX_Control(pDX, IDC_TestFileBox, m_testFileBox);
	DDX_Control(pDX, IDC_MAIN_LEARNING, editMainLearning);
	DDX_Control(pDX, IDC_MAIN_MOMENTUM, editMainMomentum);
	DDX_Control(pDX, IDC_SPACE_LEARNING, editSpaceLearning);
	DDX_Control(pDX, IDC_SPACE_MOMENTUM, editSpaceMomentum);
}

BEGIN_MESSAGE_MAP(CMBackPropDlg, CDialog)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDTRAIN, OnTrain)
	ON_BN_CLICKED(IDSTOP, OnStop)
	ON_NOTIFY(TCN_SELCHANGE, IDC_TAB, OnSelchangeTab)
	ON_WM_CLOSE()
	ON_WM_SIZE()
	ON_WM_GETMINMAXINFO()
	ON_BN_CLICKED(IDRANDOMIZEWEIGHTS, OnRandomizeWeights)
	ON_BN_CLICKED(IDLOADWEIGHTS, OnLoadWeights)
	ON_BN_CLICKED(IDSAVEWEIGHTS, OnSaveWeights)
	ON_BN_CLICKED(IDCCODE, OnGenerateCCode)
	ON_BN_CLICKED(IDLOADNETWORK, OnLoadNetwork)
	ON_BN_CLICKED(IDSAVENETWORK, OnSaveNetwork)
	ON_BN_CLICKED(IDLEARNINGMOMENTUM, OnLearningMomentum)
	ON_BN_CLICKED(IDVIEWWEIGHTS, OnViewWeights)
	ON_BN_CLICKED(IDSENSIVITY, OnSensivity)
	//}}AFX_MSG_MAP
END_MESSAGE_MAP()

/**
 Method   : BOOL CMBackPropDlg::OnInitDialog()
 Purpose  : Initialize the dialog
 Version  : 1.0.0
 Comments : If focus is set to any control the function should return FALSE
*/
BOOL CMBackPropDlg::OnInitDialog() {
	CDialog::OnInitDialog();

	if (tabs.GetItemCount() == 0) {
		// Add "About..." menu item to system menu.

		// IDM_ABOUTBOX must be in the system command range.
		ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
		ASSERT(IDM_ABOUTBOX < 0xF000);

		CMenu* pSysMenu = GetSystemMenu(FALSE);
		if (pSysMenu != NULL) {
			BOOL bNameValid;
			CString strAboutMenu;
			bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
			ASSERT(bNameValid);
			if (!strAboutMenu.IsEmpty()) {
				pSysMenu->AppendMenu(MF_SEPARATOR);
				pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
			}
		}

		// Set the icon for this dialog.
		SetIcon(m_hIcon, TRUE);	 // Set big icon
		SetIcon(m_hIcon, FALSE); // Set small icon

		tabs.InsertItem(tabTopology   , L"Topology");
		tabs.InsertItem(tabRMS        , L"RMS");
		tabs.InsertItem(tabTrainOutDes, L"Output vs Desired (training data)");
		tabs.InsertItem(tabTestOutDes , L"Output vs Desired (testing data)");

		// Determine the rectangle availiable for drawing in the tabs area.
		RECT r, rAux;
		tabs.GetClientRect(&r);
		tabs.GetItemRect(tabTopology, &rAux);
		r.left += 7;
		r.right -= 7;
		r.top = rAux.bottom + 7;
		r.bottom -=7;

		// make sure the tabs control can receive focus when the user presses tab.
		tabs.ModifyStyleEx(0, WS_EX_CONTROLPARENT);

		// Create the BPToplogy control
		MBPTopologyCtrl = new MBPTopologyControl(this);
		MBPTopologyCtrl->Create(r, &tabs);

		#ifdef MBP_WITH_CUDA
			MBPTopologyCtrl->SetCudaRestrictions((useCuda) ? 1 : 0);
		#endif

		RMSGraphic = new Graphic();		
		RMSGraphic->Create(NULL, WS_CHILD, r, &tabs, 0, NULL, FALSE);
		RMSGraphic->HorizontalAxe(_TEXT("Epoch"), 0.0, true);
		RMSGraphic->SetConsiderPreviousScale(FALSE);

		// Create the combo for the output vs desired graphics
		comboOutput	= new ComboOutput();
		comboOutput->Create(r, &tabs, false);
		comboOutput->SetFont(GetFont());
		comboOutput->GetWindowRect(&rAux);
		r.top += (rAux.bottom - rAux.top) + 3;

		// Create the output vs desired (Training Data) graphic control
		trainingOutputGraphic = new Graphic();
		trainingOutputGraphic->Create(NULL, WS_CHILD, r, &tabs, 0, NULL, FALSE);
		trainingOutputGraphic->HorizontalAxe(_TEXT("Pattern"), 1.0, true);
		trainingOutputGraphic->SetConsiderPreviousScale(FALSE);

		// Create the output vs desired (Testing Data) graphic control
		testingOutputGraphic = new Graphic();
		testingOutputGraphic->Create(NULL, WS_CHILD, r, &tabs, 0, NULL, FALSE);
		testingOutputGraphic->HorizontalAxe(_TEXT("Pattern"), 1.0, true);
		testingOutputGraphic->SetConsiderPreviousScale(FALSE);

		editMainLearning.SetValue(mainNetLearningMomentumInformation.learningRate.value);
		editMainMomentum.SetValue(mainNetLearningMomentumInformation.momentum.value);
		editSpaceLearning.SetValue(spaceNetLearningMomentumInformation.learningRate.value);
		editSpaceMomentum.SetValue(spaceNetLearningMomentumInformation.momentum.value);

		GetClientRect(&r);
		OnSize(SIZE_RESTORED, r.right, r.bottom);
	}

	srand((unsigned) clock()); // randomize

	CString commandLine = AfxGetApp()->m_lpCmdLine;

	if (!commandLine.IsEmpty()) {
		if (LoadNetwork(commandLine)) LoadTrainingTestDataIfPossible(); 
	}

	return TRUE;  // return TRUE  unless you set the focus to a control
}

/**
 Method  : void CMBackPropDlg::OnSysCommand(UINT nID, LPARAM lParam)
 Purpose : Show the about dialog when the users selects about ...
 Version : 1.0.0
*/
void CMBackPropDlg::OnSysCommand(UINT nID, LPARAM lParam) {
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)	{
		CDialog dlgAbout(IDD_ABOUTBOX);
		dlgAbout.DoModal();
	} else {
		CDialog::OnSysCommand(nID, lParam);
	}
}

/**
 Method  : void CMBackPropDlg::OnPaint()
 Purpose : Paint the Back-Propagation dialog window.
 Version : 1.0.0
*/
void CMBackPropDlg::OnPaint() {
	if (IsIconic()) {
		CPaintDC dc(this); // device context for painting

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// Center icon in client rectangle
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);

		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width()  - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;
		
		dc.DrawIcon(x, y, m_hIcon); // Draw the icon
	}	else {
		CDialog::OnPaint();
	}
}

/**
 Method  : HCURSOR CMBackPropDlg::OnQueryDragIcon()
 Purpose : The system calls this to obtain the cursor to 
		   display while the user drags the minimized window.
 Version : 1.0.0
*/
HCURSOR CMBackPropDlg::OnQueryDragIcon() {
	return static_cast<HCURSOR>(m_hIcon);
}

/**
 Method   : UINT CMBackPropDlg::TrainNetwork(LPVOID pParam)
 Purpose  : Train the network.
 Version  : 1.1.0
 Comments : This is a static method.
*/
UINT CMBackPropDlg::TrainNetwork(LPVOID pParam) {
	CMBackPropDlg * bpDlg = (CMBackPropDlg *) pParam;

	#ifdef MBP_WITH_CUDA

	if(bpDlg->useCuda) {
		bpDlg->mbpCuda = new CudaMultipleBackPropagation(bpDlg->mbp, bpDlg->trainInputPatterns, bpDlg->trainDesiredOutputPatterns);
		if (bpDlg->rmsInterval < 8) bpDlg->rmsInterval = 8;
	}

	#endif

	while (!bpDlg->stopTrain) {
		#ifdef MBP_WITH_CUDA
		
		if (bpDlg->useCuda) {
			bpDlg->TrainOneEpochUsingCuda();
		} else {
			bpDlg->TrainOneEpoch();
		}

		#else
			bpDlg->TrainOneEpoch();
		#endif
	}

	bpDlg->trainingThread = NULL;

	return 0;
}

#ifdef MBP_WITH_CUDA

void CMBackPropDlg::TrainOneEpochUsingCuda() {
	clock_t initialTime = clock();
	clock_t epochTrainingTime = 0;

	static double last_rms;
	double space_rms = 0.0;
	double rms;
	
	CString s;

	if (epoch == 0) last_rms = 1.0;

	++epoch;

	mbpCuda->Train((CUDA_FLOATING_TYPE) mbp->GetMomentum(), (CUDA_FLOATING_TYPE) mbp->GetSpaceMomentum(), robustLearning, (CUDA_FLOATING_TYPE) rmsGrowToApplyRobustLearning, (CUDA_FLOATING_TYPE) robustFactor);

	rms = mbpCuda->GetRMS();

	epochTrainingTime += clock() - initialTime;

	if (rms == 1.0) {
		rms = last_rms;
	} else if (rms != last_rms) {
		s.Format(_TEXT("%1.10f"), rms);
		trainRMSDisplay.SetWindowText(s);
		last_rms = rms;

		s.Format(L"%d", epoch);
		epochDisplay.SetWindowText(s);
	}

	if ((epoch % rmsInterval) == 0) {
		int p = epoch / rmsInterval;

		if (p == maxPointsRMS) {
			rmsInterval <<= 1;
			p >>= 1;
			for (int pos = 2; pos < maxPointsRMS; pos += 2) {
				rmsTrain[pos >> 1] = rmsTrain[pos];
				if (mbp->HasSpaceNetwork()) rmsSpaceTrain[pos >> 1] = rmsSpaceTrain[pos];
				if (rmsTest.Lenght() > 0) {
					rmsTest[pos >> 1]  = rmsTest[pos];
					if (mbp->HasSpaceNetwork()) rmsSpaceTest[pos >> 1] = rmsSpaceTest[pos];
				}
			}
		}

		int actualTab = tabs.GetCurFocus();

		rmsTrain[p] = rms;
		rmsSpaceTrain[p] = space_rms;

		//if (testVariables.Number() != 0) TestRMS(rmsTest[p], rmsSpaceTest[p], updateScreen);
				
		if (actualTab == tabRMS /*&& updateScreen*/) RMSGraphic->SetNumberPointsDraw(p + 1, (double) rmsInterval);
	}

	if (epochsStop && epoch >= numberEpochsToStop) {
		OnStop();
	} else if (!epochsStop && rms <= rmsStop) {
		OnStop();
	}

	if (stopTrain) {
		clock_t time = trainingTime / CLOCKS_PER_SEC;
		int hours = time / 3600;
		time %= 3600;

		s.Format(_TEXT("%d in %dh:%dm:%ds"), epoch, hours, time / 60, time % 60);
		epochDisplay.SetWindowText(s);


		mbpCuda->CopyNetworkHost(mbp);		
		mbpCuda = NULL;


		TrainRMS(rms, space_rms, true);

		double testRMS, SpaceRMS;
		if (testVariables.Number() != 0) TestRMS(testRMS, SpaceRMS, true);

		s.Format(_TEXT("%1.10f"), rms);
		trainRMSDisplay.SetWindowText(s);
		s.Format(_TEXT("%1.10f"), space_rms);
		trainSpaceRMSDisplay.SetWindowText(s);

		switch (tabs.GetCurFocus()) {
			case tabRMS :
				RMSGraphic->SetNumberPointsDraw((epoch == 0) ? 0 : (epoch / rmsInterval + 1), (double) rmsInterval);
				break;

			case tabTrainOutDes :
				trainingOutputGraphic->Invalidate();
				break;

			case tabTestOutDes :
				testingOutputGraphic->Invalidate();
		}

		trainButton.EnableWindow(TRUE);
		EnableOperations(TRUE);
	}
}

#endif


/**
 Method   : void TrainOneEpoch()
 Purpose  : Train one epoch.
 Version  : 1.0.0
*/
void CMBackPropDlg::TrainOneEpoch() {
	clock_t initialTime = clock();
	clock_t epochTrainingTime = 0;	

	static double last_rms = 1.0;
	double rms, space_rms, new_rms, new_space_rms;
	double newMomentum, newLearning;
	CString s;

	int inputs = mbp->Inputs();

	bool changeLearningOrMomentum = ((++epoch % 300) == 0 && (autoUpdateMomentum || autoUpdateLearning));
	if (changeLearningOrMomentum || robustLearning) mbp->KeepState();    

	mbp->Train(trainInputPatterns, trainDesiredOutputPatterns);

	epochTrainingTime += clock() - initialTime;

	mbp->CalculateRMS(rms, space_rms, trainInputPatterns, trainDesiredOutputPatterns, &networkTrainOutputs);
	initialTime = clock();
	
	if (robustLearning) {
		int t;
		for (t = 0; rms > last_rms * rmsGrowToApplyRobustLearning && t < 3; t++) {
			mbp->ReTrain(trainInputPatterns, trainDesiredOutputPatterns, robustFactor, true);
			if (updateScreen) editMainLearning.SetValue(mbp->GetLearningRate());
			epochTrainingTime += clock() - initialTime;
			mbp->CalculateRMS(rms, space_rms, trainInputPatterns, trainDesiredOutputPatterns, &networkTrainOutputs);
			initialTime = clock();
		}
		
		if (t && changeLearningOrMomentum) mbp->KeepState();
	}

	if (changeLearningOrMomentum) {
		bool updateSpaceNet = false;
		if (mbp->HasSpaceNetwork() && space_rms > mbp->minRMS) {
			if (autoUpdateLearning && autoUpdateMomentum) {
				if ((epoch % 900) == 0 || (epoch % 1200) == 0) updateSpaceNet = true;
			} else {
				if ((epoch % 600) == 0) updateSpaceNet = true;
			}
		}

		if (!autoUpdateLearning || (autoUpdateMomentum && (epoch % 600) == 0)) {
			if (updateSpaceNet) { // update space net momentum
				double momentum = mbp->GetSpaceMomentum();

				newMomentum = momentum * (1.0 - percentIncDecSpaceMomentum);
				mbp->SetSpaceMomentum(newMomentum);
				mbp->ReTrain(trainInputPatterns, trainDesiredOutputPatterns);
				epochTrainingTime += clock() - initialTime;
				mbp->CalculateRMS(new_rms, new_space_rms, trainInputPatterns, trainDesiredOutputPatterns, &networkTrainOutputs);
				initialTime = clock();

				if (new_rms >= rms && momentum != 1.0) {
					newMomentum = momentum * (1.0 + percentIncDecSpaceMomentum);
					if (newMomentum > 1.0) newMomentum = 1.0;

					mbp->SetSpaceMomentum(newMomentum);
					mbp->ReTrain(trainInputPatterns, trainDesiredOutputPatterns);
					epochTrainingTime += clock() - initialTime;
					mbp->CalculateRMS(new_rms, new_space_rms, trainInputPatterns, trainDesiredOutputPatterns, &networkTrainOutputs);
					initialTime = clock();
				}

				if (new_rms < rms) {
					rms = new_rms;
					space_rms = new_space_rms;
					if (updateScreen) editSpaceMomentum.SetValue(newMomentum);
					percentIncDecSpaceMomentum += 0.001;
				} else {
					mbp->SetSpaceMomentum(momentum);
					mbp->ReTrain(trainInputPatterns, trainDesiredOutputPatterns);
					epochTrainingTime += clock() - initialTime;
					mbp->CalculateRMS(new_rms, new_space_rms, trainInputPatterns, trainDesiredOutputPatterns, &networkTrainOutputs);
					initialTime = clock();
					if (percentIncDecSpaceMomentum > 0.001) percentIncDecSpaceMomentum -= 0.001;
				}
			} else { // update main net momentum
				double momentum = mbp->GetMomentum();

				newMomentum = momentum * (1.0 - percentIncDecMomentum);
				mbp->SetMomentum(newMomentum);
				mbp->ReTrain(trainInputPatterns, trainDesiredOutputPatterns);
				epochTrainingTime += clock() - initialTime;
				mbp->CalculateRMS(new_rms, new_space_rms, trainInputPatterns, trainDesiredOutputPatterns, &networkTrainOutputs);
				initialTime = clock();

				if (new_rms >= rms && momentum != 1.0) {
					newMomentum = momentum * (1.0 + percentIncDecMomentum);
					if (newMomentum > 1.0) newMomentum = 1.0;

					mbp->SetMomentum(newMomentum);
					mbp->ReTrain(trainInputPatterns, trainDesiredOutputPatterns);
					epochTrainingTime += clock() - initialTime;
					mbp->CalculateRMS(new_rms, new_space_rms, trainInputPatterns, trainDesiredOutputPatterns, &networkTrainOutputs);
					initialTime = clock();
				}

				if (new_rms < rms) {
					rms = new_rms;
					space_rms = new_space_rms;
					if (updateScreen) editMainMomentum.SetValue(newMomentum);
					percentIncDecMomentum += 0.001;
				} else {
					mbp->SetMomentum(momentum);
					mbp->ReTrain(trainInputPatterns, trainDesiredOutputPatterns);
					epochTrainingTime += clock() - initialTime;
					mbp->CalculateRMS(new_rms, new_space_rms, trainInputPatterns, trainDesiredOutputPatterns, &networkTrainOutputs);
					initialTime = clock();
					if (percentIncDecMomentum> 0.001) percentIncDecMomentum -= 0.001;
				}
			}
		} else { 
			if (updateSpaceNet) { // update space net learning rate
				double learning = mbp->GetSpaceLearningRate();

				newLearning = learning * (1.0 - percentIncDecSpaceLearnRate);
				mbp->SetSpaceLearningRate(newLearning);
				mbp->ReTrain(trainInputPatterns, trainDesiredOutputPatterns);
				epochTrainingTime += clock() - initialTime;
				mbp->CalculateRMS(new_rms, new_space_rms, trainInputPatterns, trainDesiredOutputPatterns, &networkTrainOutputs);
				initialTime = clock();

				if (new_rms >= rms && learning != 1.0) {
					newLearning = learning * (1.0 + percentIncDecSpaceLearnRate);
					if (newLearning > 1.0) newLearning = 1.0;

					mbp->SetSpaceLearningRate(newLearning);
					mbp->ReTrain(trainInputPatterns, trainDesiredOutputPatterns);
					epochTrainingTime += clock() - initialTime;
					mbp->CalculateRMS(new_rms, new_space_rms, trainInputPatterns, trainDesiredOutputPatterns, &networkTrainOutputs);
					initialTime = clock();
				}

				if (new_rms < rms) {
					rms = new_rms;
					space_rms = new_space_rms;
					if (updateScreen) editSpaceLearning.SetValue(newLearning);
					percentIncDecSpaceLearnRate += 0.001;
				} else {
					mbp->SetSpaceLearningRate(learning);
					mbp->ReTrain(trainInputPatterns, trainDesiredOutputPatterns);
					epochTrainingTime += clock() - initialTime;
					mbp->CalculateRMS(new_rms, new_space_rms, trainInputPatterns, trainDesiredOutputPatterns, &networkTrainOutputs);
					initialTime = clock();
					if (percentIncDecSpaceLearnRate > 0.001) percentIncDecSpaceLearnRate -= 0.001;
				}
			} else { // update main net learning rate
				double learning = mbp->GetLearningRate();

				newLearning = learning * (1.0 - percentIncDecLearnRate);
				mbp->SetLearningRate(newLearning);
				mbp->ReTrain(trainInputPatterns, trainDesiredOutputPatterns);
				epochTrainingTime += clock() - initialTime;
				mbp->CalculateRMS(new_rms, new_space_rms, trainInputPatterns, trainDesiredOutputPatterns, &networkTrainOutputs);
				initialTime = clock();

				if (new_rms >= rms && learning != 1.0) {
					newLearning = learning * (1.0 + percentIncDecLearnRate);
					if (newLearning > 1.0) newLearning = 1.0;

					mbp->SetLearningRate(newLearning);
					mbp->ReTrain(trainInputPatterns, trainDesiredOutputPatterns);
					epochTrainingTime += clock() - initialTime;
					mbp->CalculateRMS(new_rms, new_space_rms, trainInputPatterns, trainDesiredOutputPatterns, &networkTrainOutputs);
					initialTime = clock();
				}

				if (new_rms < rms) {
					rms = new_rms;
					space_rms = new_space_rms;
					if (updateScreen) editMainLearning.SetValue(newLearning);
					percentIncDecLearnRate += 0.001;
				} else {
					mbp->SetLearningRate(learning);
					mbp->ReTrain(trainInputPatterns, trainDesiredOutputPatterns);
					epochTrainingTime += clock() - initialTime;
					mbp->CalculateRMS(new_rms, new_space_rms, trainInputPatterns, trainDesiredOutputPatterns, &networkTrainOutputs);
					initialTime = clock();
					if (percentIncDecLearnRate > 0.001) percentIncDecLearnRate -= 0.001;
				}
			}			
		}
	}

	if (updateScreen) {
		s.Format(_TEXT("%1.10f"), rms);
		trainRMSDisplay.SetWindowText(s);
		s.Format(_TEXT("%1.10f"), space_rms);
		trainSpaceRMSDisplay.SetWindowText(s);
		if (tabs.GetCurSel() == tabTrainOutDes) trainingOutputGraphic->Invalidate();
	}

	if ((epoch % rmsInterval) == 0) {
		int p = epoch / rmsInterval;

		if (p == maxPointsRMS) {
			rmsInterval <<= 1;
			p >>=	1;
			for (int pos = 2; pos < maxPointsRMS; pos += 2) {
				rmsTrain[pos >> 1] = rmsTrain[pos];
				if (mbp->HasSpaceNetwork()) rmsSpaceTrain[pos >> 1] = rmsSpaceTrain[pos];
				if (rmsTest.Lenght() > 0) {
					rmsTest[pos >> 1]  = rmsTest[pos];
					if (mbp->HasSpaceNetwork()) rmsSpaceTest[pos >> 1] = rmsSpaceTest[pos];
				}
			}
		}

		int actualTab = tabs.GetCurFocus();

		rmsTrain[p] = rms;
		rmsSpaceTrain[p] = space_rms;

		if (testVariables.Number() != 0) TestRMS(rmsTest[p], rmsSpaceTest[p], updateScreen);
				
		if (actualTab == tabRMS && updateScreen) RMSGraphic->SetNumberPointsDraw(p+1, (double) rmsInterval);
	} else if (updateScreen) {
		double testRMS, SpaceRMS;

		if (testVariables.Number() != 0) TestRMS(testRMS, SpaceRMS, true);
	}

	last_rms = rms;

	if (epochsStop && epoch >= numberEpochsToStop) {
		OnStop();
	} else if (!epochsStop && rms <= rmsStop) {
		OnStop();
	}

	if (!deltaBarDelta && !autoUpdateLearning) {
		if (mainNetLearningMomentumInformation.learningRate.decayEpochs && mainNetLearningMomentumInformation.learningRate.decayPercentage) {
			if ((epoch % mainNetLearningMomentumInformation.learningRate.decayEpochs) == 0) {
				SetLearningRate(mbp->GetLearningRate() * (1.0 - (double) mainNetLearningMomentumInformation.learningRate.decayPercentage / 100));
			}
		}

		if (mbp->HasSpaceNetwork()) {
			if (spaceNetLearningMomentumInformation.learningRate.decayEpochs && spaceNetLearningMomentumInformation.learningRate.decayPercentage) {
				if ((epoch % spaceNetLearningMomentumInformation.learningRate.decayEpochs) == 0) {
					SetSpaceLearningRate(mbp->GetSpaceLearningRate() * (1.0 - (double) spaceNetLearningMomentumInformation.learningRate.decayPercentage / 100));
				}
			}
		}
	}

	if (!autoUpdateMomentum) {
		if (mainNetLearningMomentumInformation.momentum.decayEpochs && mainNetLearningMomentumInformation.momentum.decayPercentage) {
			if ((epoch % mainNetLearningMomentumInformation.momentum.decayEpochs) == 0) {
				SetMomentum(mbp->GetMomentum() * (1.0 - (double) mainNetLearningMomentumInformation.momentum.decayPercentage / 100));
			}
		}

		if (mbp->HasSpaceNetwork()) {
			if (spaceNetLearningMomentumInformation.momentum.decayEpochs && spaceNetLearningMomentumInformation.momentum.decayPercentage) {
				if ((epoch % spaceNetLearningMomentumInformation.momentum.decayEpochs) == 0) {
					SetSpaceMomentum(mbp->GetSpaceMomentum() * (1.0 - (double) spaceNetLearningMomentumInformation.momentum.decayPercentage / 100));
				}
			}
		}
	}

	epochTrainingTime += clock() - initialTime;
	trainingTime += epochTrainingTime;

	if (stopTrain) {
		clock_t time = trainingTime / CLOCKS_PER_SEC;
		int hours = time / 3600;
		time %= 3600;

		s.Format(_TEXT("%d in %dh:%dm:%ds"), epoch, hours, time / 60, time % 60);
		epochDisplay.SetWindowText(s);

		if (!updateScreen) {
			double testRMS, SpaceRMS;

			if (testVariables.Number() != 0) TestRMS(testRMS, SpaceRMS, true);

			s.Format(_TEXT("%1.10f"), rms);
			trainRMSDisplay.SetWindowText(s);
			s.Format(_TEXT("%1.10f"), space_rms);
			trainSpaceRMSDisplay.SetWindowText(s);

			switch (tabs.GetCurFocus()) {
				case tabRMS :
					RMSGraphic->SetNumberPointsDraw((epoch == 0) ? 0 : (epoch / rmsInterval + 1), (double) rmsInterval);
					break;

				case tabTrainOutDes :
					trainingOutputGraphic->Invalidate();
					break;

				case tabTestOutDes :
					testingOutputGraphic->Invalidate();
			}
		}

		trainButton.EnableWindow(TRUE);
		EnableOperations(TRUE);
	} else if (updateScreen) {
		s.Format(_TEXT("%d"), epoch);
		epochDisplay.SetWindowText(s);
	} else if (epoch % 100 == 0) {
		s.Format(_TEXT("%d"), epoch);
		epochDisplay.SetWindowText(s);
		s.Format(_TEXT("%1.10f"), rms);
		trainRMSDisplay.SetWindowText(s);
		s.Format(_TEXT("%1.10f"), space_rms);
		trainSpaceRMSDisplay.SetWindowText(s);

		double testRMS, SpaceRMS;

		if (testVariables.Number() != 0) TestRMS(testRMS, SpaceRMS, true);
	}	
}

/**
 Method   : bool CMBackPropDlg::TopologyIsValid()
 Purpose  : Returns true if the network topology 
			is valid and false otherwise.
 Version  : 1.1.0
 Comments : If the topology is invalid the user is informed 
			and the focus is set to the BPTopology control.
*/
bool CMBackPropDlg::TopologyIsValid() {
	if (MBPTopologyCtrl->GetLayers(true) < 2) {
		WarnUser(_TEXT("Invalid topology, the network should have at least an input and an output layer."));
		tabs.SetCurFocus(tabTopology);
		SelectTopologyTab();		
		return false;
	}

	return true;
}

/**
 Method   : bool CMBackPropDlg::CreateNetwork()
 Purpose  : Create the network. Returns true if 
			sucessfull otherwise returns false.
 Version  : 1.2.1
 Comments : The network will only be created if it does not exist.
*/
bool CMBackPropDlg::CreateNetwork() {
	if (!mbp.IsNull()) {
		bool variablesWithMissingValuesChanged = false;

		int inputs = mbp->Inputs();
		for(int i = 0; i < inputs; i++) {
			if (mbp->InputCanHaveMissingValues(i) != trainVariables.HasMissingValues(i)) {
				variablesWithMissingValuesChanged = true;
				break;
			}
		}

		if (variablesWithMissingValuesChanged) {
			ResetNetwork();
		} else {
			return true;
		}
	}

	if (!TopologyIsValid()) return false;

	try {
		int layers = MBPTopologyCtrl->GetLayers(true);
		Array<int> layersSize(layers);
		Array<int> neuronsWithSelectiveActivation(layers);
		List< Array<activation_function> > activationFunction;
		List< Array<double> > activationFunctionParameter;
				
		for (int l = 0; l < layers; l++) {
			int neurons = MBPTopologyCtrl->GetNeurons(l, true);
			layersSize[l] = neurons;
			neuronsWithSelectiveActivation[l] = MBPTopologyCtrl->GetNeuronsWithSelectiveActivation(l);
			Array<activation_function> * actFunct = new Array<activation_function>(neurons);
			Array<double> * actFunctParam = new Array<double>(neurons);
			for (int n = 0; n < neurons; n++) {
				(*actFunct)[n] = (activation_function) MBPTopologyCtrl->GetActivationFunction(l, n, true);
				(*actFunctParam)[n] = MBPTopologyCtrl->GetActivationFunctionParameter(l, n, true);
			}
			activationFunction.Add(actFunct);
			activationFunctionParameter.Add(actFunctParam);
		}

		int layersSpaceNetwork = MBPTopologyCtrl->GetLayers(false);

		Array<int> layersSizeSpaceNet(layersSpaceNetwork);
		List< Array<activation_function> > activationFunctionSpaceNet;
		List< Array<double> > activationFunctionParameterSpaceNet;

		if (layersSpaceNetwork > 1) {
			for (int l = 0; l < layersSpaceNetwork; l++) {
				int neurons = MBPTopologyCtrl->GetNeurons(l, false);
				layersSizeSpaceNet[l] = neurons;
				Array<activation_function> * actFunct = new Array<activation_function>(neurons);
				Array<double> * actFunctParam = new Array<double>(neurons);
				for (int n = 0; n < neurons; n++) {
					(*actFunct)[n] = (activation_function) MBPTopologyCtrl->GetActivationFunction(l, n, false);
					(*actFunctParam)[n] = MBPTopologyCtrl->GetActivationFunctionParameter(l, n, false);
				}	
				activationFunctionSpaceNet.Add(actFunct);
				activationFunctionParameterSpaceNet.Add(actFunctParam);
			}
		}

		int inputs = layersSize[0];

		Array<bool> inputMissingValues;
		inputMissingValues.Resize(inputs);
		for(int i = 0; i < inputs; i++) {
			inputMissingValues[i] = (i < trainVariables.Number()) ? trainVariables.HasMissingValues(i) : false;
		}

		mbp = new MultipleBackPropagation(layersSize, activationFunction, activationFunctionParameter, MBPTopologyCtrl->GetConnectInputLayerWithOutputLayer(TRUE), layersSizeSpaceNet, activationFunctionSpaceNet, activationFunctionParameterSpaceNet, MBPTopologyCtrl->GetConnectInputLayerWithOutputLayer(FALSE), neuronsWithSelectiveActivation, inputMissingValues);
		return true;
	} catch (BasicException e) {
		e.MakeSureUserIsInformed();
		return false;
	}
}

bool CMBackPropDlg::LoadTrainingData(bool warnUser) {
	bool successfull;

	SetCursor(AfxGetApp()->LoadStandardCursor(IDC_WAIT));

	CString trainFileName =	m_trainFileBox.GetFileName();

	try {
		if (trainFileName.IsEmpty()) throw BasicException("You must specify the training file.");

		CFileStatus fileStatus;
		if (!CFile::GetStatus(trainFileName, fileStatus)) {
			// Replace the path
			int pathSeparator = trainFileName.ReverseFind('\\');
			if (trainFileName.ReverseFind('/') > pathSeparator) pathSeparator = trainFileName.ReverseFind('/');
			if (pathSeparator == -1) pathSeparator = trainFileName.Find(':');

			trainFileName = path + trainFileName.Mid(pathSeparator + 1);
			if (CFile::GetStatus(trainFileName, fileStatus)) m_trainFileBox.SetFileName(trainFileName);
		}

		// will only read the data if the associated file has changed since it was first readed.
		if (lastTimeTrainingDataWasModified != fileStatus.m_mtime) {
			lastTimeTrainingDataWasModified = fileStatus.m_mtime;
			reloadTrainingData = true;
		}

		if (reloadTrainingData) {
			if (!warnUser) trainVariables.RetriesBeforeThrowExceptions(0);
			trainVariables.Read(trainFileName);
			reloadTrainingData = false;
		}
		successfull = true;
	} catch (BasicException e) {
		if (warnUser) {
			if (!e.UserWasInformed()) WarnUser(e.Cause(), L"Invalid Train File");
			m_trainFileBox.SetFocus();
		}
		successfull = false;
	}

	if (!warnUser) trainVariables.RetriesBeforeThrowExceptions(HandleExceptions::DefaultRetriesBeforeThrowExceptions());

	if (successfull) {
		int inputs = MBPTopologyCtrl->GetInputs();
		int outputs = MBPTopologyCtrl->GetOutputs();

		if (trainVariables.Number() == inputs + outputs) {
			int trainPatterns = trainVariables.Columns();

			trainInputPatterns.Resize(trainPatterns, inputs);

			for (int i = 0; i < inputs; i++) { // input variables will be rescaled between -1 and 1
				double min = trainVariables.Minimum(i);
				double max = trainVariables.Maximum(i);

				for (int p = 0; p < trainPatterns; p++) {
					double value = trainVariables.Value(i, p);

					if (_isnan(value)) {
						trainInputPatterns[p][i] = MISSING_VALUE;
					} else {
						trainInputPatterns[p][i] = (min == max) ? 1.0 : (-1.0 + 2.0 * (value - min) / (max - min));
					}
				}
			}

			trainDesiredOutputPatterns.Resize(trainPatterns, outputs);

			desiredTrainOutputs.Resize(outputs, trainPatterns);
			trainingOutputGraphic->SetNumberPointsDraw(desiredTrainOutputs.Columns(), 1);

			networkTrainOutputs.Resize(outputs, trainPatterns);

			int outputLayer = MBPTopologyCtrl->GetLayers(true) - 1;
			int spaceOutputLayer = MBPTopologyCtrl->GetLayers(false) - 1;

			int outputNeuronsWithSelectiveActivation = MBPTopologyCtrl->GetNeuronsWithSelectiveActivation(outputLayer);

			for (int o = 0; o < outputs; o++) {
				int outVar = o + inputs;

				/*
				// Calculate the number of different values for this variable.
				double values[3];
				long numberDifferentValues = 0;
				for (int p = 0; p < trainPatterns; p++) {
					double value = trainVariables.Value(outVar, p);
					for (int v = 0; v < numberDifferentValues; v++) if (values[v] == value) break;
					if (v > numberDifferentValues) {
						values[numberDifferentValues] = value;
						if (++numberDifferentValues > 3) break;					
					}
				}
				*/

				// Determine the new minimum and maximum for this variable
				activation_function a = (activation_function) MBPTopologyCtrl->GetActivationFunction(outputLayer, o, true);

				/*
				double newMin, newMax;
				if (numberDifferentValues > 3) {
					newMin = (a == Tanh || a == Linear) ? -0.9 : 0.1;
					newMax = 0.9;
				} else {
					newMin = (a == Tanh || a == Linear) ? -1.0 : 0.0;
					newMax = 1.0;
				}
				*/
				double newMin = (a == Tanh || a == Linear) ? -1.0 : 0.0;

				/*if (o < outputNeuronsWithSelectiveActivation && newMin != -1.0) {
					a = (activation_function) BPTopologyCtrl->GetActivationFunction(spaceOutputLayer, o, false);
					if (a == Tanh || a == Linear) newMin = -1.0;
				}*/

				trainVariables.newMinimum[outVar] = newMin;
				//trainVariables.newMaximum[outVar] = newMax;

				double min = trainVariables.Minimum(outVar);
				double max = trainVariables.Maximum(outVar);

				for (int p = 0; p < trainPatterns; p++) {
					double value = trainVariables.Value(outVar, p);

					if (_isnan(value)) {
						trainDesiredOutputPatterns[p][o] = desiredTrainOutputs[o][p] = MISSING_VALUE;
					} else {
						trainDesiredOutputPatterns[p][o] = desiredTrainOutputs[o][p] = (min == max) ? 1.0 : (newMin + (value - min) * (1.0 - newMin) / (max - min));
					}
				}
			}
		}
	}

	SetCursor(AfxGetApp()->LoadStandardCursor(IDC_ARROW));

	return successfull;
}

bool CMBackPropDlg::LoadTestingData(bool warnUser) {
	bool successfull = true;

	CString testFileName = m_testFileBox.GetFileName();

	if (testFileName.IsEmpty()) {
		testVariables.Clear();

		testingOutputGraphic->Clear();
		testingOutputGraphic->SetNumberPointsDraw(0, 1);

		testDesiredOutputPatterns.Resize(0, 0);
		desiredTestOutputs.Resize(0, 0);
		networkTestOutputs.Resize(0, 0);
	} else {		
		CFileStatus fileStatus;
		if (!CFile::GetStatus(testFileName, fileStatus)) {
			// Replace the path
			int pathSeparator = testFileName.ReverseFind('\\');
			if (testFileName.ReverseFind('/') > pathSeparator) pathSeparator = testFileName.ReverseFind('/');
			if (pathSeparator == -1) pathSeparator = testFileName.Find(':');

			testFileName = path + testFileName.Mid(pathSeparator + 1);
			if (CFile::GetStatus(testFileName, fileStatus)) m_testFileBox.SetFileName(testFileName);			
		}

		// will only read the data if the associated file has changed since it was first readed.
		if (lastTimeTestingDataWasModified != fileStatus.m_mtime) {
			lastTimeTestingDataWasModified = fileStatus.m_mtime;
			reloadTestingData = true;
		}

		if (reloadTestingData) {
			try {
				if (!warnUser) testVariables.RetriesBeforeThrowExceptions(0);
				testVariables.Read(testFileName);
				reloadTestingData = false;
			} catch (BasicException e) {
				if (warnUser) {
					if (!e.UserWasInformed()) WarnUser(e.Cause(), L"Invalid Test File");
					m_testFileBox.SetFocus();
				}
				successfull = false;
			}
		}
	}

	if (!warnUser) testVariables.RetriesBeforeThrowExceptions(HandleExceptions::DefaultRetriesBeforeThrowExceptions());

	if (successfull) {
		int inputs = MBPTopologyCtrl->GetInputs();
		int outputs = MBPTopologyCtrl->GetOutputs();

		if (testVariables.Number() ==  inputs + outputs) {
			int testPatterns = testVariables.Columns();

			testInputPatterns.Resize(testPatterns, inputs);

			if (trainVariables.Number() != testVariables.Number()) {
				WarnUser(_TEXT("Please note that since the training data could not be loaded, the testing data may not be properly scaled."));
			}

			for (int i = 0; i < inputs; i++) { // input variables will be rescaled between -1 and 1
				double min, max;

				if (trainVariables.Number() == testVariables.Number()) {
					min = trainVariables.Minimum(i);
					max = trainVariables.Maximum(i);
				} else {
					min = testVariables.Minimum(i);
					max = testVariables.Maximum(i);
				}

				for (int p = 0; p < testPatterns; p++) { 
					double value = testVariables.Value(i, p);

					if (_isnan(value)) {
						testInputPatterns[p][i] = MISSING_VALUE;
					} else {
						testInputPatterns[p][i] = (min == max) ? 1 : (- 1 + (2 * (value - min)) / (max - min));
					}
				}
			}

			testDesiredOutputPatterns.Resize(testPatterns, outputs);

			desiredTestOutputs.Resize(outputs, testPatterns);
			testingOutputGraphic->SetNumberPointsDraw(desiredTestOutputs.Columns(), 1);

			networkTestOutputs.Resize(outputs, testPatterns);

			int outputLayer = MBPTopologyCtrl->GetLayers(true) - 1;

			for (int o = 0; o < outputs; o++) {
				int outVar = o + inputs;

				double newMin, min, max;

				if (trainVariables.Number() == testVariables.Number()) {
					newMin = testVariables.newMinimum[outVar] = trainVariables.newMinimum[outVar];

					min = trainVariables.Minimum(outVar);
					max = trainVariables.Maximum(outVar);
	
				} else {
					// Determine the new minimum and maximum for this variable
					activation_function a = (activation_function) MBPTopologyCtrl->GetActivationFunction(outputLayer, o, true);

					newMin = testVariables.newMinimum[outVar] = (a == Tanh || a == Linear) ? -1.0 : 0.0;

					min = testVariables.Minimum(outVar);
					max = testVariables.Maximum(outVar);
				}

				for (int p = 0; p< testPatterns; p++) {
					double value = testVariables.Value(outVar, p);

					if (_isnan(value)) {
						testDesiredOutputPatterns[p][o] = desiredTestOutputs[o][p] = MISSING_VALUE;
					} else {
						testDesiredOutputPatterns[p][o] = desiredTestOutputs[o][p] = (min == max) ? 1.0 : (newMin + (value - min) * (1.0 - newMin) / (max - min));
					}
				}
			}
		}
	}

	return successfull;
}

bool CMBackPropDlg::NetworkIsValid(bool checkCudaAndMissingValues) {
	int inputs = MBPTopologyCtrl->GetInputs();
	int outputs = MBPTopologyCtrl->GetOutputs();

	if (!LoadTrainingData()) return false;
	if (trainVariables.Number() !=  inputs + outputs) {
		WarnUser(L"The number of network inputs plus the number of outputs does not match the number of variables in the training file.");
		MBPTopologyCtrl->SetFocus();
		tabs.SetCurFocus(tabTopology);
		return false;
	}

	if (!LoadTestingData()) return false;
	if (testVariables.Number() != 0) {
		if (trainVariables.Number() != testVariables.Number()) {
			WarnUser(_TEXT("The number of network inputs plus the number of outputs does not match the number of variables in the testing file."));
			m_testFileBox.SetFocus();
			return false;
		}
	}

	for (int o = 0; o < outputs; o++) {
		int outVar = o + inputs;

		bool hasMissingValues = false;
		CString dataset;

		if (trainVariables.HasMissingValues(outVar)) {
			hasMissingValues = true;
			if (!testVariables.HasMissingValues(outVar)) dataset = L" (training dataset)";
		} else if (outVar < testVariables.Number() && testVariables.HasMissingValues(outVar)) {
			hasMissingValues = true;
			dataset = L" (test dataset)";
		}

		if (hasMissingValues) {
			CString columnName = trainVariables.Name(outVar);
			if (columnName.IsEmpty()) {
				columnName.Format(L"%d", outVar + 1);
			} else {
				columnName = L"«" + columnName + L"»";
			}

			WarnUser(L"Column " + columnName + L" has missing values" + dataset + L". Output columns cannot have missing values.");
			MBPTopologyCtrl->SetFocus();
			tabs.SetCurFocus(tabTopology);
			return false;
		}
	}

	int columnsWithMissingValues = 0;
	for (int i = 0; i < inputs; i++) {
		if (trainVariables.HasMissingValues(i)) {
			columnsWithMissingValues++;
		} else if (i < testVariables.Number() && testVariables.HasMissingValues(i)) {
			CString columnName = trainVariables.Name(i);
			if (columnName.IsEmpty()) {
				columnName.Format(L"%d", i + 1);
			} else {
				columnName = L"«" + columnName + L"»";
			}

			WarnUser(L"Column " + columnName + L" has missing values in test dataset but not in the training dataset. A variable can only have missing values in the test dataset if it has also missing values on the training dataset.");
			MBPTopologyCtrl->SetFocus();
			tabs.SetCurFocus(tabTopology);
			return false;
		}
	}

	/*if (columnsWithMissingValues && checkCudaAndMissingValues && useCuda) {
		WarnUser(L"Currently missing values are only supported by CUDA. Please disable CUDA if you want to train a network with missing values.");
		return false;
	}*/

	return true;
}

/**
 Method   : void OnTrain()
 Puropse  : Train the network.
 Called when the user clicks the train button
*/
void CMBackPropDlg::OnTrain() {
	if (!NetworkIsValid(true)) return;

	LoadTrainingTestDataIfPossible();

	if (!CreateNetwork()) return;

	trainButton.EnableWindow(FALSE);
	EnableOperations(FALSE);

	FillComboOutput(trainVariables);
	OutputChanged(0);

	rmsTrain.Resize(maxPointsRMS);
	rmsSpaceTrain.Resize(maxPointsRMS);
	stopTrain = true;

	if (epoch == 0)	{
		double rms, spaceRMS;

		TrainRMS(rms, spaceRMS, true);
		rmsTrain[0] = rms;
		rmsSpaceTrain[0] = spaceRMS;

		percentIncDecLearnRate = percentIncDecSpaceLearnRate = 0.01;
		percentIncDecMomentum = percentIncDecSpaceMomentum = 0.01;
	}

	/*TrainRMS(lastRMS, lastSpaceRMS);
	if (epoch == 0)	{
		rmsTrain[0] = lastRMS;
		rmsSpaceTrain[0] = lastSpaceRMS;
		lastRMSVariantion = 0.0;
	}

	if (lastRMSVariantion == 0.0) {
		lastSpaceRMSVariantion = 0.0;
		numberEpochsBeforeUpdateLearningMomentum = 100;
		percentIncDecLearnRate = percentIncDecSpaceLearnRate = 0.01;
		percentIncDecMomentum = percentIncDecSpaceMomentum = 0.01;
		increasingLearningRate = increasingSpaceLearningRate = true;
		increasingMomentum = increasingSpaceMomentum = true;

		updateLearning = true;
		updateSpaceLearning = true;
	}*/

	RMSGraphic->Clear();
	RMSGraphic->InsertLine(rmsTrain.Pointer(), _TEXT("Training"));

	if (testVariables.Number() != 0) {
		if (epoch == 0) {
			rmsTest.Resize(maxPointsRMS);
			rmsSpaceTest.Resize(maxPointsRMS);
			TestRMS(rmsTest[0], rmsSpaceTest[0], true);
		} else if (rmsTest.Lenght() == 0) {
			rmsTest.Resize(maxPointsRMS);
			rmsSpaceTest.Resize(maxPointsRMS);
			for (int p = 0; p < maxPointsRMS; p++) rmsTest[p] = rmsSpaceTest[p] = 0.0;
		}

		RMSGraphic->InsertLine(rmsTest.Pointer(), _TEXT("Testing"));
	} else {
		rmsTest.Resize(0);
		testRMSDisplay.SetWindowText(_TEXT("0.0000000000"));
		testSpaceRMSDisplay.SetWindowText(_TEXT("0.0000000000"));
	}

	if (mbp->HasSpaceNetwork()) {
		RMSGraphic->InsertLine(rmsSpaceTrain.Pointer(), _TEXT("Space Network Training"));
		if (testVariables.Number() != 0) RMSGraphic->InsertLine(rmsSpaceTest.Pointer(), _TEXT("Space Network Testing"));
	} else {
		trainSpaceRMSDisplay.SetWindowText(_TEXT("0.0000000000"));
		testSpaceRMSDisplay.SetWindowText(_TEXT("0.0000000000"));
	}
	
	if (epoch == 0)	rmsInterval = 1;

	tabs.SetCurFocus(tabRMS);
	RMSGraphic->Invalidate();

	mbp->SetDeltaBarDelta(deltaBarDelta);
	mbp->BatchTraining(batchTraining);
	mbp->SetRandomPatternPresentation(randomizePatterns);

	if (epoch == 0 || !deltaBarDelta) {
		mbp->SetLearningRate(editMainLearning.GetValue());
		mbp->SetSpaceLearningRate(editSpaceLearning.GetValue());
	}

	mbp->SetMomentum(editMainMomentum.GetValue());
	mbp->SetSpaceMomentum(editSpaceMomentum.GetValue());

	mbp->SetWeightDecayFactor(weightDecay);

	mbp->minRMS = spaceRmsStop;

	stopTrain = false;	

	SetPriorityClass(GetCurrentProcess(), programPriority);

	trainingThread = AfxBeginThread(TrainNetwork, this, THREAD_PRIORITY_TIME_CRITICAL);

	stopButton.EnableWindow(TRUE);
}

/**
 Method   : void CMBackPropDlg::EnableOperations(BOOL value)
 Purpose  : Enable or prevent user from doing operations.
 Version  : 1.0.0
*/
void CMBackPropDlg::EnableOperations(BOOL value) {
	m_trainFileBox.SetEnabled(value);
	m_testFileBox.SetEnabled(value);
	MBPTopologyCtrl->SetEnabled(value);
	randomizeWeightsButton.EnableWindow(value);
	saveWeightsButton.EnableWindow(value);
	loadWeightsButton.EnableWindow(value);
	LearningMomentum.EnableWindow(value);
	generateCCode.EnableWindow(value);
	loadNetwork.EnableWindow(value);
	saveNetwork.EnableWindow(value);
	viewWeights.EnableWindow(value);
	sensitivityButton.EnableWindow(value);
}

/**
 Method   : void CMBackPropDlg::OnStop()
 Purpose  : Stop training.
 Version  : 1.0.0
*/
void CMBackPropDlg::OnStop() {
	stopButton.EnableWindow(FALSE);
	StopTraining();
}

/**
 Method   : void CMBackPropDlg::SelectTopologyTab()
 Purpose  : Selects the topology tab
 Version  : 1.0.0
*/
void CMBackPropDlg::SelectTopologyTab() {
	MBPTopologyCtrl->ShowWindow(SW_SHOW);
	RMSGraphic->ShowWindow(SW_HIDE);
	comboOutput->ShowWindow(SW_HIDE);
	trainingOutputGraphic->ShowWindow(SW_HIDE);
	testingOutputGraphic->ShowWindow(SW_HIDE);
	MBPTopologyCtrl->SetFocus();
}

/**
 Method   : void CMBackPropDlg::OnSelchangeTab(NMHDR * pNMHDR, LRESULT * pResult)
 Purpose  : Show and hide the appropriate controls according 
			to the tab selected. Set the focus to one of the 
						shown controls if they can receive focus.
 Version  : 1.1.0
*/
void CMBackPropDlg::OnSelchangeTab(NMHDR * pNMHDR, LRESULT * pResult) {
	int tab = tabs.GetCurFocus();

	switch (tab) {
		case tabTopology :
			SelectTopologyTab();
			break;

		case tabRMS :
			MBPTopologyCtrl->ShowWindow(SW_HIDE);
			RMSGraphic->SetNumberPointsDraw((epoch == 0) ? 0 : (epoch / rmsInterval + 1), (double) rmsInterval);
			RMSGraphic->ShowWindow(SW_SHOW);
			comboOutput->ShowWindow(SW_HIDE);
			trainingOutputGraphic->ShowWindow(SW_HIDE);
			testingOutputGraphic->ShowWindow(SW_HIDE);
			break;

		case tabTrainOutDes :
			MBPTopologyCtrl->ShowWindow(SW_HIDE);
			RMSGraphic->ShowWindow(SW_HIDE);

			if (trainingThread == NULL) {
				if (reloadTestingData) LoadTrainingTestDataIfPossible(false);
				reloadTrainingData = true;
			}

			testingOutputGraphic->ShowWindow(SW_HIDE);

			#ifdef MBP_WITH_CUDA
			if (trainingThread == NULL || !useCuda)
			#endif
			{
				comboOutput->ShowWindow(SW_SHOW);
				comboOutput->SetFocus();
				trainingOutputGraphic->ShowWindow(SW_SHOW);
				trainingOutputGraphic->Invalidate();
			}
			break;

		case tabTestOutDes :
			MBPTopologyCtrl->ShowWindow(SW_HIDE);
			RMSGraphic->ShowWindow(SW_HIDE);

			if (trainingThread == NULL) {
				if (reloadTestingData || reloadTrainingData) {
					LoadTrainingTestDataIfPossible(false);
					if (tabs.GetCurFocus() == tabTopology) break;
				}
			}

			trainingOutputGraphic->ShowWindow(SW_HIDE);

			#ifdef MBP_WITH_CUDA
			if (trainingThread == NULL || !useCuda)
			#endif
			{
				comboOutput->ShowWindow(SW_SHOW);
				comboOutput->SetFocus();
				testingOutputGraphic->ShowWindow(SW_SHOW);
				testingOutputGraphic->Invalidate();
			}
	}

	*pResult = 0;
}

/**
 Method   : void CMBackPropDlg::OnClose()
 Purpose  : Make sure the training thread stops 
			execution before closing the application.
 Version  : 1.0.0
*/
void CMBackPropDlg::OnClose() {
	if (trainingThread != NULL) {
		trainingThread->Delete();
		trainingThread = NULL;		
	}

	CDialog::OnClose();
}

/**
 Method   : void CMBackPropDlg::OnSize(UINT nType, int cx, int cy)
 Purpose  : Resize all the windows inside the dialog,
			each time the dialog is sized.
 Version  : 1.0.0
*/
void CMBackPropDlg::OnSize(UINT nType, int cx, int cy) {
	CDialog::OnSize(nType, cx, cy);

	if (trainButton.GetSafeHwnd() != NULL) {
		CRect r, rAux;
		int x, width, left, xfTabs;

		// Move the train button
		trainButton.GetWindowRect(r);
		ScreenToClient(r);
		width = r.Width();
		r.right = cx - 3;
		r.left = cx - 3 - width;
		trainButton.MoveWindow(r);

		// Move the stop button
		stopButton.GetWindowRect(r);
		ScreenToClient(r);
		r.right = cx - 3;
		x = r.left = cx - 3 - width;
		stopButton.MoveWindow(r);

		// Move the data files frame
		frameDataFiles.GetWindowRect(r);
		ScreenToClient(r);
		x = r.right = x - 3;
		frameDataFiles.MoveWindow(r);

		// Move the train file box
		m_trainFileBox.GetWindowRect(r);
		ScreenToClient(r);
		r.right = x - 7;
		m_trainFileBox.MoveWindow(r);

		// Move the test file box
		m_testFileBox.GetWindowRect(r);
		ScreenToClient(r);
		r.right = x - 7;
		m_testFileBox.MoveWindow(r);

		// Move the epoch frame
		frameEpoch.GetWindowRect(r);
		ScreenToClient(r);
		width = r.Width();
		x = r.right = cx - 3;
		r.left = cx - 3 - width;
		xfTabs = r.left - 3;
		frameEpoch.MoveWindow(r);

		// Move the epoch display
		epochDisplay.GetWindowRect(r);
		ScreenToClient(r);
		width = r.Width();
		r.right = x - 7;
		r.left = x - 7 - width;
		epochDisplay.MoveWindow(r);

		// Move the learning frame
		frameLearning.GetWindowRect(r);
		ScreenToClient(r);
		width = r.Width();
		x = r.right = cx - 3;
		left = r.left = cx - 3 - width;
		frameLearning.MoveWindow(r);

		// Move the main network learning frame
		frameMainLearning.GetWindowRect(r);
		ScreenToClient(r);
		left += (width - r.Width()) / 2;
		width = r.Width();
		r.left = left;
		r.right = r.left + width;		
		frameMainLearning.MoveWindow(r);

		// Move the space network learning frame
		frameSpaceLearning.GetWindowRect(r);
		ScreenToClient(r);
		r.left = left;
		x = r.right = r.left + width;		
		frameSpaceLearning.MoveWindow(r);

		// Move the main network learning edit box
		editMainLearning.GetWindowRect(r);
		ScreenToClient(r);
		width = r.Width();
		r.right = x - 7;
		r.left = x - 7 - width;
		editMainLearning.MoveWindow(r);

		// Move the main network momentum edit box
		editMainMomentum.GetWindowRect(r);
		ScreenToClient(r);
		width = r.Width();
		r.right = x - 7;
		r.left = x - 7 - width;
		editMainMomentum.MoveWindow(r);

		// Move the space network learning edit box
		editSpaceLearning.GetWindowRect(r);
		ScreenToClient(r);
		width = r.Width();
		r.right = x - 7;
		r.left = x - 7 - width;
		editSpaceLearning.MoveWindow(r);

		// Move the space network momentum edit box
		editSpaceMomentum.GetWindowRect(r);
		ScreenToClient(r);
		width = r.Width();
		r.right = x - 7;
		x = r.left = x - 7 - width;
		editSpaceMomentum.MoveWindow(r);

		// Move the main network learning label
		labelMainLearning.GetWindowRect(r);
		ScreenToClient(r);
		width = r.Width();
		r.right = x - 3;
		x = r.left = x - 3 - width;
		labelMainLearning.MoveWindow(r);

		// Move the main network momentum label
		labelMainMomentum.GetWindowRect(r);
		ScreenToClient(r);
		width = r.Width();
		r.left = x;
		r.right = x + width;
		labelMainMomentum.MoveWindow(r);

		// Move the space network learning label
		labelSpaceLearning.GetWindowRect(r);
		ScreenToClient(r);
		width = r.Width();
		r.left = x;
		r.right = x + width;
		labelSpaceLearning.MoveWindow(r);

		// Move the space network momentum label
		labelSpaceMomentum.GetWindowRect(r);
		ScreenToClient(r);
		width = r.Width();
		r.left = x;
		r.right = x + width;
		labelSpaceMomentum.MoveWindow(r);

		// Move the RMS frame
		frameRMS.GetWindowRect(r);
		ScreenToClient(r);
		width = r.Width();
		r.right = cx - 3;
		left = r.left = cx - 3 - width;
		frameRMS.MoveWindow(r);

		// Move the main RMS frame
		frameMainRMS.GetWindowRect(r);
		ScreenToClient(r);
		left += (width - r.Width()) / 2;
		width = r.Width();
		r.left = left;
		r.right = r.left + width;
		frameMainRMS.MoveWindow(r);

		// Move the space RMS frame
		frameSpaceRMS.GetWindowRect(r);
		ScreenToClient(r);
		r.left = left;
		x = r.right = r.left + width;		
		frameSpaceRMS.MoveWindow(r);

		// Move the main train RMS display
		trainRMSDisplay.GetWindowRect(r);
		ScreenToClient(r);
		width = r.Width();
		r.right = x - 7;
		r.left = x - 7 - width;
		trainRMSDisplay.MoveWindow(r);

		// Move the main test RMS display
		testRMSDisplay.GetWindowRect(r);
		ScreenToClient(r);
		r.right = x - 7;
		r.left = x - 7 - width;
		testRMSDisplay.MoveWindow(r);

		// Move the space train RMS display
		trainSpaceRMSDisplay.GetWindowRect(r);
		ScreenToClient(r);
		width = r.Width();
		r.right = x - 7;
		r.left = x - 7 - width;
		trainSpaceRMSDisplay.MoveWindow(r);

		// Move the space test RMS display
		testSpaceRMSDisplay.GetWindowRect(r);
		ScreenToClient(r);
		r.right = x - 7;
		x = r.left = x - 7 - width;
		testSpaceRMSDisplay.MoveWindow(r);

		// Move the main label of the train RMS display
		labelTrainRMSDisplay.GetWindowRect(r);
		ScreenToClient(r);
		width = r.Width();
		r.right = x - 3;
		r.left = x - 3 - width;
		labelTrainRMSDisplay.MoveWindow(r);
		
		// Move the main label of the test RMS display
		labelTestRMSDisplay.GetWindowRect(r);
		ScreenToClient(r);
		width = r.Width();
		r.right = x - 3;
		r.left = x - 3 - width;
		labelTestRMSDisplay.MoveWindow(r);

		// Move the space label of the train RMS display
		labelSpaceTrainRMSDisplay.GetWindowRect(r);
		ScreenToClient(r);
		width = r.Width();
		r.right = x - 3;
		r.left = x - 3 - width;
		labelSpaceTrainRMSDisplay.MoveWindow(r);
		
		// Move the space label of the test RMS display
		labelSpaceTestRMSDisplay.GetWindowRect(r);
		ScreenToClient(r);
		width = r.Width();
		r.right = x - 3;
		r.left = x - 3 - width;
		labelSpaceTestRMSDisplay.MoveWindow(r);

		// Move the network frame
		frameNetwork.GetWindowRect(r);
		ScreenToClient(r);
		width = r.Width();
		r.right = cx - 3;
		r.left = cx - 3 - width;
		frameNetwork.MoveWindow(r);

		// Move the weights frame
		frameWeights.GetWindowRect(r);
		ScreenToClient(r);
		width = r.Width();
		r.right = cx - 3;
		left = r.left = cx - 3 - width;
		frameWeights.MoveWindow(r);

		// Move the randomize (weights) button
		randomizeWeightsButton.GetWindowRect(r);
		ScreenToClient(r);
		left += (width - r.Width()) / 2;
		width = r.Width();
		r.left = left;
		r.right = left + width;
		randomizeWeightsButton.MoveWindow(r);

		// Move the view (weights) button
		viewWeights.GetWindowRect(r);
		ScreenToClient(r);
		r.left = left;
		r.right = left + width;
		viewWeights.MoveWindow(r);

		// Move the load (weights) button
		loadWeightsButton.GetWindowRect(r);
		ScreenToClient(r);
		r.left = left;
		r.right = left + width;
		loadWeightsButton.MoveWindow(r);

		// Move the save (weights) button
		saveWeightsButton.GetWindowRect(r);
		ScreenToClient(r);
		r.left = left;
		r.right = left + width;
		saveWeightsButton.MoveWindow(r);

		// Move the generate C code button
		generateCCode.GetWindowRect(r);
		ScreenToClient(r);
		r.left = left;
		r.right = left + width;
		generateCCode.MoveWindow(r);

		// Move the load (network) button
		loadNetwork.GetWindowRect(r);
		ScreenToClient(r);
		r.left = left;
		r.right = left + width;
		loadNetwork.MoveWindow(r);

		// Move the save (network) button
		saveNetwork.GetWindowRect(r);
		ScreenToClient(r);
		r.left = left;
		r.right = left + width;
		saveNetwork.MoveWindow(r);

		// Move the (network) input sensitivity button
		sensitivityButton.GetWindowRect(r);
		ScreenToClient(r);
		r.left = left;
		r.right = left + width;
		sensitivityButton.MoveWindow(r);

		// Move the Learning & Momentum Adjust (configure) button
		LearningMomentum.GetWindowRect(r);
		ScreenToClient(r);
		r.left = left;
		r.right = left + width;
		LearningMomentum.MoveWindow(r);

		// Move the tabs separator control
		tabs.GetWindowRect(r);
		ScreenToClient(r);
		r.right = xfTabs;
		r.bottom = cy - 3;
		tabs.MoveWindow(r);

		// Move the controls inside the tabs control.
		tabs.GetClientRect(&r);
		tabs.GetItemRect(tabTopology, &rAux);
		r.left += 7;
		r.right -= 7;
		r.top = rAux.bottom + 7;
		r.bottom -=7;

		MBPTopologyCtrl->MoveWindow(r);
		RMSGraphic->MoveWindow(r);
		comboOutput->MoveWindow(r);

		r.top += (rAux.bottom - rAux.top) + 3;

		trainingOutputGraphic->MoveWindow(r);
		testingOutputGraphic->MoveWindow(r);

		Invalidate();
	}
}

/**
 Method  : void CMBackPropDlg::UpdateLearningMomentum()
 Purpose : Update the values of the learning rate and 
			   momentum for the main and the space network.
 Version : 1.1.0
*/
void CMBackPropDlg::UpdateLearningMomentum() {
	double learning = mainNetLearningMomentumInformation.learningRate.value;
	double momentum = mainNetLearningMomentumInformation.momentum.value;
	double spaceLearning = spaceNetLearningMomentumInformation.learningRate.value;
	double spaceMomentum = spaceNetLearningMomentumInformation.momentum.value;

	/*if (!autoUpdateLearning) {
		long percentageDecay = mainNetLearningMomentumInformation.learningRate.decayPercentage;
		int epochDecay = mainNetLearningMomentumInformation.learningRate.decayEpochs;
		for (int epoch = this->epoch; epochDecay && epoch >= epochDecay; epoch -= epochDecay) learning *= (1.0 - (double) percentageDecay / 100);
		
		percentageDecay = spaceNetLearningMomentumInformation.learningRate.decayPercentage;
		epochDecay = spaceNetLearningMomentumInformation.learningRate.decayEpochs;
		for (epoch = this->epoch; epochDecay && epoch >= epochDecay; epoch -= epochDecay) spaceLearning *= (1.0 - (double) percentageDecay / 100);
	}

	if (!autoUpdateMomentum) {
		long percentageDecay = mainNetLearningMomentumInformation.momentum.decayPercentage;
		int epochDecay = mainNetLearningMomentumInformation.momentum.decayEpochs;
		for (epoch = this->epoch; epochDecay && epoch >= epochDecay; epoch -= epochDecay) momentum *= (1.0 - (double) percentageDecay / 100);

		percentageDecay = spaceNetLearningMomentumInformation.momentum.decayPercentage;
		epochDecay = spaceNetLearningMomentumInformation.momentum.decayEpochs;
		for (epoch = this->epoch; epochDecay && epoch >= epochDecay; epoch -= epochDecay) spaceMomentum *= (1.0 - (double) percentageDecay / 100);
	}*/

	editMainLearning.SetValue(learning);
	editMainMomentum.SetValue(momentum);
	editSpaceLearning.SetValue(spaceLearning);
	editSpaceMomentum.SetValue(spaceMomentum);
}

/**
 Method   : void CMBackPropDlg::OnGetMinMaxInfo(MINMAXINFO FAR* lpMMI)
 Purpose  : Establish the minimum size of the Back-Propagation dialog.
 Version  : 1.0.0
*/
void CMBackPropDlg::OnGetMinMaxInfo(MINMAXINFO FAR* lpMMI) {
	if (frameLearning.GetSafeHwnd() != NULL) {
		CRect r;

		frameLearning.GetWindowRect(r);

		lpMMI->ptMinTrackSize.x = 1000; 
		lpMMI->ptMinTrackSize.y = 750;
	}
	
	CDialog::OnGetMinMaxInfo(lpMMI);
}

BEGIN_EVENTSINK_MAP(CMBackPropDlg, CDialog)
  //{{AFX_EVENTSINK_MAP(CMBackPropDlg)
	ON_EVENT(CMBackPropDlg, IDC_TestFileBox,  1 /* Change */, OnChangeTestFileBox , VTS_NONE)
	ON_EVENT(CMBackPropDlg, IDC_TrainFileBox, 1 /* Change */, OnChangeTrainFileBox, VTS_NONE)
	//}}AFX_EVENTSINK_MAP
END_EVENTSINK_MAP()

/**
 Method   : void CMBackPropDlg::OnChangeTestFileBox()
 Purpose  : Indicate that the testing data needs to be reloaded.
 Version  : 1.0.0
*/
void CMBackPropDlg::OnChangeTestFileBox() {
	reloadTestingData = true;
}

/**
 Method   : void CMBackPropDlg::OnChangeTrainFileBox()
 Purpose  : Indicate that the training data needs to be reloaded.
 Version  : 1.0.0
*/
void CMBackPropDlg::OnChangeTrainFileBox() {
	reloadTrainingData = true;
}

/**
 Method   : void CMBackPropDlg::OnRandomizeWeights()
 Purpose  : Randomize the network weights between two 
			values determined by the user. 
 Comments : This is done only if the topology is valid.
			Called when the user clicks the randomize weights button.
*/
void CMBackPropDlg::OnRandomizeWeights() {
	LoadTrainingTestDataIfPossible();
	if (mbp.IsNull()) return;

	RandWeightsDialog dialog(this);

	if (dialog.DoModal() == IDOK) {
		epochDisplay.SetWindowText(L"0");
		RMSGraphic->Clear();
		ResetNetwork();
		LoadTrainingTestDataIfPossible();
	}
}

/**
 Method   : void CMBackPropDlg::OnLoadWeights()
 Purpose  : Load network weights.
 Called when the user clicks the load weights button
*/
void CMBackPropDlg::OnLoadWeights() {
	LoadTrainingTestDataIfPossible();
	if (mbp.IsNull()) return;

	LoadWeightsDialog dialog(this);
	if (dialog.DoModal() == IDOK) {
		epoch = 0;
		trainingTime = 0;
		UpdateLearningMomentum();
		LoadTrainingTestDataIfPossible();
	}
}

/**
 Method   : void CMBackPropDlg::OnSaveWeights()
 Purpose  : Load network weights.
 Called when the user clicks the save weights button
*/
void CMBackPropDlg::OnSaveWeights() {
	LoadTrainingTestDataIfPossible();
	if (mbp.IsNull()) return;

	SaveWeightsDialog dialog(this);
	dialog.DoModal();
}

/**
 Method   : void CMBackPropDlg::OnGenerateCCode()
 Purpose  : Generate C code for the network.
 Called when the user clicks the generate C code button
*/
void CMBackPropDlg::OnGenerateCCode() {
	LoadTrainingTestDataIfPossible();
	if (mbp.IsNull()) return;

	GenerateCCodeDialog dialog(this);
	dialog.DoModal();
}

/**
 Method   : void CMBackPropDlg::OnLoadNetwork()
 Purpose  : Load a network.
 Called when the user clicks the load network button
*/
void CMBackPropDlg::OnLoadNetwork() {
	LoadNetworkDialog dialog(this);
	if (dialog.DoModal() == IDOK) LoadTrainingTestDataIfPossible();
}

/**
 Method   : void CMBackPropDlg::OnSaveNetwork()
 Purpose  : Save a network.
 Called when the user clicks the save network button
*/
void CMBackPropDlg::OnSaveNetwork() {
	LoadTrainingTestDataIfPossible();
	if (mbp.IsNull()) return;

	SaveNetworkDialog dialog(this);
	dialog.DoModal();
}

/**
Load training and test data if possible. 
Calculate the corresponding RMS errors and update graphics. 
If the network was not created yet it will be created.
*/
void CMBackPropDlg::LoadTrainingTestDataIfPossible(bool changeTab) {
	double rms, spaceRMS;
	bool comboFilled = false;		
	int tab;

	bool trainingDataLoaded = LoadTrainingData(false);
	bool testDataLoaded = LoadTestingData(false);

	int selOutput = comboOutput->GetCurSel();
	
	comboOutput->ResetContent();

	if (mbp.IsNull()) {
		ResetNetwork();
		epochDisplay.SetWindowText(_TEXT("0"));
		RMSGraphic->Clear();

		if (!CreateNetwork()) return;
	}

	int numberVariables = mbp->Inputs() + mbp->Outputs();

	#ifdef MBP_WITH_CUDA
		mbpCuda = NULL;
	#endif

	if (trainingDataLoaded) {
		if (trainVariables.Number() == numberVariables) {
			TrainRMS(rms, spaceRMS, true);
			tab = tabTrainOutDes;
			FillComboOutput(trainVariables);
			comboFilled = true;
		} else {
			trainRMSDisplay.SetWindowText(_TEXT("1.0000000000"));
			trainSpaceRMSDisplay.SetWindowText(_TEXT("1.0000000000"));
		}
	}

	if (testDataLoaded) {
		if (testVariables.Number() == numberVariables) {
			TestRMS(rms, spaceRMS, true);
			tab = tabTestOutDes;
			if (!comboFilled) {
				FillComboOutput(testVariables);
				comboFilled = true;
			}
		} else {
			testRMSDisplay.SetWindowText((testVariables.Number() == 0) ? _TEXT("0.0000000000") : _TEXT("1.0000000000"));
		}
	}	

	if (comboFilled) {
		if (selOutput < 0 || selOutput >= mbp->Outputs()) selOutput = 0;
		OutputChanged(selOutput);
		if (changeTab) tabs.SetCurFocus(tab);
	}
}

/**
 Method   : void CMBackPropDlg::FillComboOutput(VariablesData & v)
 Purpose  : Fullfill the output combo with the network outputs.
 Version  : 1.0.0
*/
void CMBackPropDlg::FillComboOutput(VariablesData & varsData) {
	
	comboOutput->ResetContent();

	int numberVariables = mbp->Inputs() + mbp->Outputs();

	for (int v = mbp->Inputs(); v < numberVariables; v++) {
		CString name = varsData.Name(v);
		if (name.IsEmpty()) name.Format(_TEXT("Output #%d"), v - mbp->Inputs() + 1);
		comboOutput->AddString(name);
	}
	comboOutput->SetCurSel(0);	
}

/**
 Method   : void CMBackPropDlg::TrainRMS(double & rms, double & spaceRMS)
 Purpose  : Returns the train root mean square error.
 Version  : 1.1.0
 Comments : The method also displays the rms value and invalidates 
			the training output vs desired graphic.
*/
void CMBackPropDlg::TrainRMS(double & rms, double & spaceRMS, bool display) {
	CString s;

	int inputs = mbp->Inputs();

	mbp->CalculateRMS(rms, spaceRMS, trainInputPatterns, trainDesiredOutputPatterns, &networkTrainOutputs);

	if (display) {
		s.Format(_TEXT("%1.10f"), rms);
		trainRMSDisplay.SetWindowText(s);
		s.Format(_TEXT("%1.10f"), spaceRMS);
		trainSpaceRMSDisplay.SetWindowText(s);

		if (tabs.GetCurSel() == tabTrainOutDes) trainingOutputGraphic->Invalidate();
	}
}

/**
 Method   : void CMBackPropDlg::TestRMS(double & rms, double & spaceRMS)
 Purpose  : Returns the test root mean square error.
 Version  : 1.0.0
 Comments : The method also displays the rms value and invalidates 
			the testing output vs desired graphic.
*/
void CMBackPropDlg::TestRMS(double & rms, double & spaceRMS, bool display) {
	CString s;

	int inputs = mbp->Inputs();

	mbp->CalculateRMS(rms, spaceRMS, testInputPatterns, testDesiredOutputPatterns, &networkTestOutputs);

	if (display) {
		s.Format(_TEXT("%1.10f"), rms);
		testRMSDisplay.SetWindowText(s);
		s.Format(_TEXT("%1.10f"), spaceRMS);
		testSpaceRMSDisplay.SetWindowText(s);

		if (tabs.GetCurSel() == tabTestOutDes) testingOutputGraphic->Invalidate();
	}
}

/**
 Method   : void CMBackPropDlg::OutputChanged(int newOutput)
 Purpose  : Called each time the output variable that the 
			user is seeing changes. At this point graphics 
						must change in order to reflect the changes.
 Version  : 1.0.0
*/
void CMBackPropDlg::OutputChanged(int newOutput) {
	double originalMinimum;
	double originalMaximum;
	double actualMinimum;

	int inputs = mbp->Inputs();
	int variables = inputs + mbp->Outputs();

	trainingOutputGraphic->Clear();
	if (trainVariables.Number() == variables && newOutput >= 0) {
		originalMinimum = trainVariables.Minimum(inputs + newOutput);
		originalMaximum = trainVariables.Maximum(inputs + newOutput);
		actualMinimum = trainVariables.newMinimum[inputs + newOutput];
		//double actualMaximum = trainVariables.newMaximum[inputs + newOutput];
	
		trainingOutputGraphic->InsertLine(desiredTrainOutputs[newOutput].Pointer(), _TEXT("Desired Output"));
		trainingOutputGraphic->InsertLine(networkTrainOutputs[newOutput].Pointer(), _TEXT("Network Output"));
		trainingOutputGraphic->Rescale(originalMinimum, originalMaximum, actualMinimum, 1.0);
		//trainingOutputGraphic->Rescale(originalMinimum, originalMaximum, actualMinimum, actualMaximum);
		trainingOutputGraphic->SetNumberPointsDraw(desiredTrainOutputs.Columns(), 1);
	} else if (testVariables.Number() == variables) {
		originalMinimum = testVariables.Minimum(inputs + newOutput);
		originalMaximum = testVariables.Maximum(inputs + newOutput);
		actualMinimum = testVariables.newMinimum[inputs + newOutput];
	}

	testingOutputGraphic->Clear();
	if (testVariables.Number() == variables && newOutput >= 0) {
		testingOutputGraphic->InsertLine(desiredTestOutputs[newOutput].Pointer(), _TEXT("Desired Output"));
		testingOutputGraphic->InsertLine(networkTestOutputs[newOutput].Pointer(), _TEXT("Network Output"));
		testingOutputGraphic->Rescale(originalMinimum, originalMaximum, actualMinimum, 1.0);
		//testingOutputGraphic->Rescale(originalMinimum, originalMaximum, actualMinimum, actualMaximum);
		testingOutputGraphic->SetNumberPointsDraw(desiredTestOutputs.Columns(), 1);
	}
}

/**
 Method   : void CMBackPropDlg::StopTraining()
 Purpose  : If the network is trainig stop the training.
 Version  : 1.1.0
*/
void CMBackPropDlg::StopTraining() {
	stopTrain = true;
}

/**
 Method   : bool CMBackPropDlg::GenerateCCode(CString & filename)
 Purpose  : Generate C code for the network and 
			save it into the given filename.
 Version  : 1.0.0
*/
bool CMBackPropDlg::GenerateCCode(CString & filename) {
	try {
		OutputFile f(filename);
		mbp->GenerateCCode(f, trainVariables, MBPTopologyCtrl->GetConnectInputLayerWithOutputLayer(TRUE), MBPTopologyCtrl->GetConnectInputLayerWithOutputLayer(FALSE));
		return true;
	} catch (BasicException e) {
		e.MakeSureUserIsInformed();
		return false;
	}
}

/**
 Method   : bool CMBackPropDlg::SaveWeights(CString & filename)
 Purpose  : Save the network weights. Returns 
			true if succefull, false otherwise. 
 Version  : 1.0.0
*/
bool CMBackPropDlg::SaveWeights(CString & filename) {
	try {
		CString MBPVersion;
		MBPVersion.LoadString(IDS_VERSION);

		CString s;

		OutputFile f(filename);

		f.WriteLine("Multiple Back-Propagation Weights");
		f.WriteLine(MBPVersion);
		f.WriteLine(" Multiple Back-Propagation can be freely obtained at http://dit.ipg.pt/MBP");

		// Save the main network topology
		int layers = MBPTopologyCtrl->GetLayers(true);
		s.Format(_TEXT("%d"), layers);
		f.WriteLine(s);

		for (int l = 0; l < layers; l++) {
			int neurons = MBPTopologyCtrl->GetNeurons(l, true);
			s.Format(_TEXT("%d"), neurons);
			f.WriteLine(s);
		}

		// Save the space network topology
		layers = MBPTopologyCtrl->GetLayers(false);
		s.Format(_TEXT("%d"), layers);
		f.WriteLine(s);

		for (int l = 0; l < layers; l++) {
			int neurons = MBPTopologyCtrl->GetNeurons(l, false);
			s.Format(_TEXT("%d"), neurons);
			f.WriteLine(s);
		}

		mbp->SaveWeights(f);
		mbp->SaveSpaceWeights(f);
		return true;
	} catch (BasicException e) {
		e.MakeSureUserIsInformed();
		return false;
	}
}

/**
 Method   : bool CMBackPropDlg::LoadWeights(CString & filename)
 Purpose  : Load the network weights. Returns 
			true if succefull, false otherwise. 
 Version  : 1.0.0
*/
bool CMBackPropDlg::LoadWeights(CString & filename) {
	try {
		CString s;			

		InputFile f(filename);

		if (!f.ReadLine(s) || s != "Multiple Back-Propagation Weights") throw BasicException(unrecognizedFormat);

		if (!f.ReadLine(s) || s.Left(34) != "Multiple Back-Propagation Version ") throw BasicException(unrecognizedFormat);

		if (!f.ReadLine(s)) throw BasicException(unrecognizedFormat); // comment line

		if (!f.ReadLine(s)) throw BasicException(unrecognizedFormat);
		int layers = StringToInteger(s);
		if (layers != MBPTopologyCtrl->GetLayers(true)) throw BasicException("Incorrect number of layers. The file you attempt to read does not contain weights for this network.");

		for (int l = 0; l < layers; l++) {
			if (!f.ReadLine(s)) throw BasicException(unrecognizedFormat);
			int neurons = StringToInteger(s);
			if (neurons != MBPTopologyCtrl->GetNeurons(l, true)) throw BasicException("Incorrect number of neurons in layer. The file you attempt to read does not contain weights for this network.");
		}

		if (!f.ReadLine(s)) throw BasicException(unrecognizedFormat);
		layers = StringToInteger(s);
		if (layers > 0) {
			if (layers != MBPTopologyCtrl->GetLayers(false)) throw BasicException("Incorrect number of layers. The file you attempt to read does not contain weights for this network.");

			for (int l = 0; l < layers; l++) {
				if (!f.ReadLine(s)) throw BasicException(unrecognizedFormat);
				int neurons = StringToInteger(s);
				if (neurons != MBPTopologyCtrl->GetNeurons(l, false)) throw BasicException("Incorrect number of neurons in layer. The file you attempt to read does not contain weights for this network.");
			}
		}

		mbp->LoadWeights(f);
		if (layers) {
			mbp->LoadSpaceWeights(f);
		}

		return true;
	} catch (BasicException e) {
		e.MakeSureUserIsInformed();
		return false;
	}
}

/**
 Method   : bool CMBackPropDlg::SaveNetwork(CString & filename)
 Purpose  : Save the network. Returns true if succefull, false otherwise.
 Version  : 1.0.0
*/
bool CMBackPropDlg::SaveNetwork(CString & filename) {
	try {
		CString MBPVersion;
		MBPVersion.LoadString(IDS_VERSION);

		if (filename.Find('\\') == -1 && filename.Find(':') == -1) filename = path + filename;
		OutputFile f(filename);
		CString s;

		f.WriteLine(MBPVersion);
		f.WriteLine("Multiple Back-Propagation can be freely obtained at http://dit.ipg.pt/MBP");
		f.WriteLine(m_trainFileBox.GetFileName());
		f.WriteLine(m_testFileBox.GetFileName());

		s.Format(_TEXT("%ld\n"), programPriority);
		f.WriteString(s);
		f.WriteLine((updateScreen)   ? "1" : "0");

		f.WriteLine((deltaBarDelta) ? "1" : "0");
		s.Format(_TEXT("%1.15f\n%1.15f\n%1.15f\n"), Connection::u, Connection::d, Connection::maxStepSize);
		f.WriteString(s);

		f.WriteLine((robustLearning) ? "1" : "0");
		s.Format(_TEXT("%1.15f\n%1.15f\n"), robustFactor, rmsGrowToApplyRobustLearning);
		f.WriteString(s);

		s.Format(_TEXT("%1.15f\n"), weightDecay);
		f.WriteString(s);

		if (autoUpdateLearning) {
			f.WriteLine("1");
			mainNetLearningMomentumInformation.learningRate.value = editMainLearning.GetValue();
			spaceNetLearningMomentumInformation.learningRate.value = editSpaceLearning.GetValue();
		} else {
			f.WriteLine("0");
		}

		if (autoUpdateMomentum) {
			f.WriteLine("1");
			mainNetLearningMomentumInformation.momentum.value = editMainMomentum.GetValue();
			spaceNetLearningMomentumInformation.momentum.value = editSpaceMomentum.GetValue();
		} else {
			f.WriteLine("0");
		}

		s.Format(_TEXT("%1.15f\n%1.15f\n%1.15f\n%1.15f\n"), percentIncDecLearnRate, percentIncDecMomentum, percentIncDecSpaceLearnRate, percentIncDecSpaceMomentum);
		f.WriteString(s);

		s.Format(_TEXT("%1.15f\n%ld\n%ld\n"), mainNetLearningMomentumInformation.learningRate.value, mainNetLearningMomentumInformation.learningRate.decayEpochs, mainNetLearningMomentumInformation.learningRate.decayPercentage);
		f.WriteString(s);

		s.Format(_TEXT("%1.15f\n%ld\n%ld\n"), mainNetLearningMomentumInformation.momentum.value, mainNetLearningMomentumInformation.momentum.decayEpochs, mainNetLearningMomentumInformation.momentum.decayPercentage);
		f.WriteString(s);

		s.Format(_TEXT("%1.15f\n%ld\n%ld\n"), spaceNetLearningMomentumInformation.learningRate.value, spaceNetLearningMomentumInformation.learningRate.decayEpochs, spaceNetLearningMomentumInformation.learningRate.decayPercentage);
		f.WriteString(s);

		s.Format(_TEXT("%1.15f\n%ld\n%ld\n"), spaceNetLearningMomentumInformation.momentum.value, spaceNetLearningMomentumInformation.momentum.decayEpochs, spaceNetLearningMomentumInformation.momentum.decayPercentage);
		f.WriteString(s);

		f.WriteLine((epochsStop) ? "1" : "0");
		s.Format(_TEXT("%1.15f\n%d\n"), rmsStop, numberEpochsToStop);
		f.WriteString(s);

		s.Format(_TEXT("%1.15f"), spaceRmsStop);
		f.WriteLine(s);

		f.WriteLine((batchTraining) ? "1" : "0");

		f.WriteLine((randomizePatterns) ? "1" : "0");
	
		s.Format(_TEXT("%d"), MBPTopologyCtrl->GetNetworkType());
		f.WriteLine(s);

		f.WriteLine(MBPTopologyCtrl->GetText(true));
		f.WriteLine(MBPTopologyCtrl->GetText(false));

		int layers = MBPTopologyCtrl->GetLayers(true);
		for (int l = 1; l < layers; l++) {
			s.Format(_TEXT("%d"), MBPTopologyCtrl->GetNeuronsWithSelectiveActivation(l));
			f.WriteLine(s);

			int neurons = MBPTopologyCtrl->GetNeurons(l, true);
			for (int n = 0; n < neurons; n++) {
				s.Format(_TEXT("%d\n%1.15f\n"), MBPTopologyCtrl->GetActivationFunction(l, n, true), MBPTopologyCtrl->GetActivationFunctionParameter(l,n, true));
				f.WriteString(s);
			}
		}

		layers = MBPTopologyCtrl->GetLayers(false);
		for (int l = 1; l < layers; l++) {
			int neurons = MBPTopologyCtrl->GetNeurons(l, false);

			for (int n = 0; n < neurons; n++) {
				s.Format(_TEXT("%d\n%1.15f\n"), MBPTopologyCtrl->GetActivationFunction(l, n, false), MBPTopologyCtrl->GetActivationFunctionParameter(l,n, false));
				f.WriteString(s);
			}
		}

		f.WriteLine((MBPTopologyCtrl->GetConnectInputLayerWithOutputLayer(TRUE)) ? "1" : "0");
		f.WriteLine((MBPTopologyCtrl->GetConnectInputLayerWithOutputLayer(FALSE)) ? "1" : "0");

		mbp->SaveWeights(f, "\n", true);
		mbp->SaveSpaceWeights(f, "\n", true);
		
		s.Format(_TEXT("%d\n%d\n%d\n"), epoch, rmsInterval, trainingTime);
		f.WriteString(s);

		if (epoch > 0) {
			int lenght = rmsTrain.Lenght();
			s.Format(_TEXT("%d"), lenght);
			f.WriteLine(s);

			for (int i = 0; i < lenght; i++) {
				s.Format(_TEXT("%1.15f"), rmsTrain[i]);
				f.WriteLine(s);
			}

			lenght = rmsTest.Lenght();
			s.Format(_TEXT("%d"), lenght);
			f.WriteLine(s);

			for (int i = 0; i < lenght; i++) {
				s.Format(_TEXT("%1.15f"), rmsTest[i]);
				f.WriteLine(s);
			}

			lenght = (mbp->HasSpaceNetwork()) ? rmsSpaceTrain.Lenght() : 0;
			s.Format(_TEXT("%d"), lenght);
			f.WriteLine(s);

			for (int i = 0; i < lenght; i++) {
				s.Format(_TEXT("%1.15f"), rmsSpaceTrain[i]);
				f.WriteLine(s);
			}

			lenght = (mbp->HasSpaceNetwork()) ? rmsSpaceTest.Lenght() : 0;
			s.Format(_TEXT("%d"), lenght);
			f.WriteLine(s);

			for (int i = 0; i < lenght; i++) {
				s.Format(_TEXT("%1.15f"), rmsSpaceTest[i]);
				f.WriteLine(s);
			}
		}

		SetWindowTitle(filename);

		return true;
	} catch (BasicException e) {
		e.MakeSureUserIsInformed();
		return false;
	}
}

/**
 Method   : bool CMBackPropDlg::LoadNetwork(CString & filename)
 Purpose  : Load the network. Returns true if succefull, false otherwise. 
 Version  : 1.0.0
*/
bool CMBackPropDlg::LoadNetwork(CString & filename) {
	try {
		if (filename.Find('\\') == -1 && filename.Find(':') == -1) filename = path + filename;
		InputFile f(filename);
		CString s;

		if (!f.ReadLine(s) || s.Left(34) != "Multiple Back-Propagation Version ") throw BasicException(unrecognizedFormat);

		CString version = s.Mid(34);

		if (version == L"2.1.0") version = "2.0.6"; // This beta version wrote files as the 2.0.6

		if (!f.ReadLine(s)) throw BasicException(unrecognizedFormat); // comment line

		LoadString(f, s);
		m_trainFileBox.SetFileName(s);
		LoadString(f, s);
		m_testFileBox.SetFileName(s);

		if (version >= _TEXT("1.6.0")) {
			if (version >= _TEXT("1.9.2")) {
				LoadString(f, s);
				if (version >= _TEXT("1.9.6")) {
					programPriority = StringToLong(s);
				} else {
					programPriority = (s == "1") ? HIGH_PRIORITY_CLASS : REALTIME_PRIORITY_CLASS;
				}

				updateScreen = (version >= _TEXT("1.9.4")) ? LoadBool(f) : false;
			} else {
				programPriority = NORMAL_PRIORITY_CLASS;
				updateScreen = true;
			}

			deltaBarDelta = LoadBool(f);
			Connection::u = LoadDouble(f);
			Connection::d = LoadDouble(f);
			Connection::maxStepSize = 1.0;
			if (version >= L"2.0.7") Connection::maxStepSize = LoadDouble(f);
			robustLearning = LoadBool(f);
			robustFactor = LoadDouble(f);
			rmsGrowToApplyRobustLearning = LoadDouble(f);
			weightDecay = LoadDouble(f);
		} else {
			deltaBarDelta = false;
			robustLearning = false;
			weightDecay = 0.0;
		}

		if (version >= _TEXT("1.2.0")) {
			bool autoUpdate = LoadBool(f);

			if (version >= _TEXT("1.6.0")) {
				autoUpdateLearning = (autoUpdate && !deltaBarDelta);
				autoUpdateMomentum = LoadBool(f);

				if (version >= _TEXT("1.6.2")) {
					percentIncDecLearnRate = LoadDouble(f);
					percentIncDecMomentum = LoadDouble(f);
					percentIncDecSpaceLearnRate = LoadDouble(f);
					percentIncDecSpaceMomentum = LoadDouble(f);
				} else {
					percentIncDecLearnRate = percentIncDecMomentum = percentIncDecSpaceLearnRate = percentIncDecSpaceMomentum = 0.01;
				}
			} else {
				autoUpdateLearning = autoUpdateMomentum = autoUpdate;
				percentIncDecLearnRate = percentIncDecMomentum = percentIncDecSpaceLearnRate = percentIncDecSpaceMomentum = 0.01;
			}
		} else {
			autoUpdateLearning = autoUpdateMomentum = false;
			percentIncDecLearnRate = percentIncDecMomentum = percentIncDecSpaceLearnRate = percentIncDecSpaceMomentum = 0.01;
		}

		mainNetLearningMomentumInformation.learningRate.value = LoadDouble(f);
		mainNetLearningMomentumInformation.learningRate.decayEpochs = LoadInt(f);
		mainNetLearningMomentumInformation.learningRate.decayPercentage = LoadInt(f);

		mainNetLearningMomentumInformation.momentum.value = LoadDouble(f);
		mainNetLearningMomentumInformation.momentum.decayEpochs = LoadInt(f);
		mainNetLearningMomentumInformation.momentum.decayPercentage = LoadInt(f);

		spaceNetLearningMomentumInformation.learningRate.value = LoadDouble(f);
		spaceNetLearningMomentumInformation.learningRate.decayEpochs = LoadInt(f);
		spaceNetLearningMomentumInformation.learningRate.decayPercentage = LoadInt(f);

		spaceNetLearningMomentumInformation.momentum.value = LoadDouble(f);
		spaceNetLearningMomentumInformation.momentum.decayEpochs = LoadInt(f);
		spaceNetLearningMomentumInformation.momentum.decayPercentage = LoadInt(f);

		batchTraining = false;

		if (version >= _TEXT("1.2.1")) {
			epochsStop = LoadBool(f);
			rmsStop = LoadDouble(f);
			numberEpochsToStop = LoadInt(f);

			spaceRmsStop = (version >= _TEXT("1.9.1")) ? LoadDouble(f) : 0.0;

			if (version >= _TEXT("1.3.0")) batchTraining = LoadBool(f);

			randomizePatterns = LoadBool(f);
		} else {
			epochsStop = false;
			rmsStop = 0.0;
			numberEpochsToStop = 1000000;
			randomizePatterns = false;
		}

		MBPTopology::network_type networkType = (MBPTopology::network_type) LoadInt(f);
		MBPTopologyCtrl->SetNetworkType(networkType);

		LoadString(f, s);
		MBPTopologyCtrl->SetText(s, true);

		LoadString(f, s);
		MBPTopologyCtrl->SetText(s, false);

		mbp = NULL;

		#ifdef MBP_WITH_CUDA
			mbpCuda = NULL;
		#endif

		RMSGraphic->Clear();

		#ifdef MBP_WITH_CUDA
			mbpCuda = NULL;
			useCuda = cuda.Supported() && !updateScreen && deltaBarDelta && (weightDecay == 0.0) && !autoUpdateMomentum && (mainNetLearningMomentumInformation.momentum.decayPercentage == 0) && (spaceNetLearningMomentumInformation.momentum.decayPercentage == 0) && (spaceRmsStop == 0.0) && batchTraining && (networkType != MBPHO);
		#endif

		LoadLayers(f, true);
		LoadLayers(f, false);

		if (version >= _TEXT("1.8.0")) {
			BOOL connectionsBetweenInputsAndOutputs = LoadBool(f);
			MBPTopologyCtrl->SetConnectInputLayerWithOutputLayer(TRUE, connectionsBetweenInputsAndOutputs);
			#ifdef MBP_WITH_CUDA
				if (connectionsBetweenInputsAndOutputs) useCuda = false;
			#endif
				
			connectionsBetweenInputsAndOutputs = LoadBool(f);
			MBPTopologyCtrl->SetConnectInputLayerWithOutputLayer(FALSE, connectionsBetweenInputsAndOutputs);
			#ifdef MBP_WITH_CUDA
				if (connectionsBetweenInputsAndOutputs) useCuda = false;
			#endif
		} else {
			MBPTopologyCtrl->SetConnectInputLayerWithOutputLayer(TRUE, FALSE);
			MBPTopologyCtrl->SetConnectInputLayerWithOutputLayer(FALSE, FALSE);
		}

		int pathSeparator = filename.ReverseFind('\\');
		if (filename.ReverseFind('/') > pathSeparator) pathSeparator = filename.ReverseFind('/');
		if (pathSeparator == -1) pathSeparator = filename.Find(':');

		path = filename.Left(pathSeparator + 1);

		LoadTrainingData(false);
		if (!CreateNetwork()) return false;

		mbp->LoadWeights(f, (version >= _TEXT("1.5.0")));
		mbp->LoadSpaceWeights(f, (version >= _TEXT("1.5.0")));

		epoch = LoadInt(f);
		s.Format(_TEXT("%d"), epoch);
		epochDisplay.SetWindowText(s);
		UpdateLearningMomentum();

		if (!f.ReadLine(s)) {
			epoch = 0;
			trainingTime = 0;
			UpdateLearningMomentum();
			throw BasicException(unrecognizedFormat);
		}
		
		rmsInterval = StringToInteger(s);

		if (version >= _TEXT("1.6.4")) {
			if (!f.ReadLine(s)) {
				epoch = 0;
				trainingTime = 0;
				UpdateLearningMomentum();
				throw BasicException(unrecognizedFormat);
			}
		
			trainingTime = StringToLong(s);

			clock_t time = trainingTime / CLOCKS_PER_SEC;
			int hours = time / 3600;
			time %= 3600;

			s.Format(_TEXT("%d in %dh:%dm:%ds"), epoch, hours, time / 60, time % 60);
			epochDisplay.SetWindowText(s);
		} else {
			trainingTime = 0;
		}

		if (epoch > 0) {
			if (!f.ReadLine(s)) {
				epoch = 0;
				trainingTime = 0;
				UpdateLearningMomentum();
				throw BasicException(unrecognizedFormat);
			}
			int lenght = StringToInteger(s);
			rmsTrain.Resize(lenght);

			for (int i = 0; i < lenght; i++) {
				if (!f.ReadLine(s)) {
					epoch = 0;
					trainingTime = 0;
					UpdateLearningMomentum();
					throw BasicException(unrecognizedFormat);
				}
				rmsTrain[i] = StringToDouble(s);
			}
			if (lenght > 0) RMSGraphic->InsertLine(rmsTrain.Pointer(), _TEXT("Training"));

			if (!f.ReadLine(s)) {
				epoch = 0;
				trainingTime = 0;
				UpdateLearningMomentum();
				throw BasicException(unrecognizedFormat);
			}
			lenght = StringToInteger(s);
			rmsTest.Resize(lenght);			

			for (int i = 0; i < lenght; i++) {
				if (!f.ReadLine(s)) {
					epoch = 0;
					trainingTime = 0;
					UpdateLearningMomentum();
					throw BasicException(unrecognizedFormat);
				}
				rmsTest[i] = StringToDouble(s);
			}			
			if (lenght > 0) RMSGraphic->InsertLine(rmsTest.Pointer(), _TEXT("Testing"));

			if (version >= _TEXT("1.2.2")) {
				if (!f.ReadLine(s)) {
					epoch = 0;
					trainingTime = 0;
					UpdateLearningMomentum();
					throw BasicException(unrecognizedFormat);
				}
				lenght = StringToInteger(s);
				rmsSpaceTrain.Resize(lenght);

				for (int i = 0; i < lenght; i++) {
					if (!f.ReadLine(s)) {
						epoch = 0;
						trainingTime = 0;
						UpdateLearningMomentum();
						throw BasicException(unrecognizedFormat);
					}
					rmsSpaceTrain[i] = StringToDouble(s);
				}			
				if (lenght > 0) RMSGraphic->InsertLine(rmsSpaceTrain.Pointer(), _TEXT("Space Network Training"));

				if (!f.ReadLine(s)) {
					epoch = 0;
					trainingTime = 0;
					UpdateLearningMomentum();
					throw BasicException(unrecognizedFormat);
				}
				lenght = StringToInteger(s);
				rmsSpaceTest.Resize(lenght);

				for (int i = 0; i < lenght; i++) {
					if (!f.ReadLine(s)) {
						epoch = 0;
						trainingTime = 0;
						UpdateLearningMomentum();
						throw BasicException(unrecognizedFormat);
					}
					rmsSpaceTest[i] = StringToDouble(s);
				}			
				if (lenght > 0) RMSGraphic->InsertLine(rmsSpaceTest.Pointer(), _TEXT("Space Network Testing"));
			}
		}

		SetWindowTitle(filename);

		return true;
	} catch (BasicException e) {
		e.MakeSureUserIsInformed();
		return false;
	}
}

void CMBackPropDlg::SetWindowTitle(CString & filename) {
	int pathSeparator = filename.ReverseFind('\\');
	if (filename.ReverseFind('/') > pathSeparator) pathSeparator = filename.ReverseFind('/');
	if (pathSeparator == -1) pathSeparator = filename.Find(':');

	CString version;
	version.LoadStringW(IDS_VERSION);

	SetWindowText(version + " > " + filename.Mid(pathSeparator + 1));
	path = filename.Left(pathSeparator + 1);
}

void CMBackPropDlg::LoadString(InputFile & f, CString & s) {
	if (!f.ReadLine(s)) throw BasicException(unrecognizedFormat);
}

int CMBackPropDlg::LoadInt(InputFile & f) {
	CString s;
	LoadString(f, s);

	return StringToInteger(s);
}

double CMBackPropDlg::LoadDouble(InputFile & f) {
	CString s;
	LoadString(f, s);

	return StringToDouble(s);
}

bool CMBackPropDlg::LoadBool(InputFile & f) {
	CString s;
	LoadString(f, s);

	return (s == "1");
}

void CMBackPropDlg::LoadLayers(InputFile & f, bool network) {
	int layers = MBPTopologyCtrl->GetLayers(network);
	
	for (int l = 1; l < layers; l++) {
		if (network) MBPTopologyCtrl->SetNeuronsWithSelectiveActivation(l, LoadInt(f));

		int neurons = MBPTopologyCtrl->GetNeurons(l, network);
		for (int n = 0; n < neurons; n++) {
			int af = LoadInt(f);
			MBPTopologyCtrl->SetActivationFunction(l, n, network, af);

			#ifdef MBP_WITH_CUDA
				if (af != Sigmoid) useCuda = false;
			#endif

			double k = LoadDouble(f);
			MBPTopologyCtrl->SetActivationFunctionParameter(l, n, network, k);

			#ifdef MBP_WITH_CUDA
				if (k != 1.0) useCuda = false;
			#endif
		}
	}
}

void CMBackPropDlg::OnLearningMomentum() {
	if (autoUpdateLearning) {
		mainNetLearningMomentumInformation.learningRate.value = editMainLearning.GetValue();
		spaceNetLearningMomentumInformation.learningRate.value = editSpaceLearning.GetValue();
	} 
		
	if (autoUpdateMomentum) {
		mainNetLearningMomentumInformation.momentum.value = editMainMomentum.GetValue();
		spaceNetLearningMomentumInformation.momentum.value = editSpaceMomentum.GetValue();
	}

	LearningMomentumDialog dialog(this);
	dialog.DoModal();	
}

// Called when the user clicks the view weights button
void CMBackPropDlg::OnViewWeights() {
	LoadTrainingTestDataIfPossible();
	if (mbp.IsNull()) return;

	WeightsDialog dialog(this);
	dialog.DoModal();
}

// Called when the user clicks the sensivity button
void CMBackPropDlg::OnSensivity() {
	if (!NetworkIsValid()) return;
	if (!CreateNetwork()) return;

	SensitivityDialog dialog(this);
	dialog.DoModal();
}