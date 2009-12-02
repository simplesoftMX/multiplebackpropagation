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
 Class    : CBackPropDlg
 Purpose  : Represents the Back-Propagation dialog.
 Date     : 3 of July of 1999.
 Reviewed : 5 of February of 2001.
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
                                  -->| CBackPropDlg |
                                      --------------
*/

#pragma once
#include "../Common/Files/VariablesData.h"
#include "AditionalDefinitions.h"
#include "CtrlGraphic.h"
#include "CtrlCientificNumber.h"
#include "CtrlFilebox.h"
#include "MBPTopologyCtrl.h"
#include "MultipleBackPropagation.h"
#include "MbpCuda/CudaMultipleBackPropagation.h"
#include "ComboOutput.h"
#include "MBackProp.h"
#include "afxwin.h"
#include "cuda.h"

class CMBackPropDlg : public CDialog {
	private :
		friend class LearningMomentumDialog;
		friend class WeightsDialog;
		friend class SensitivityDialog;

		enum {
			tabTopology = 0,
			tabRMS = 1,
			tabTrainOutDes = 2,
			tabTestOutDes = 3
		};

		#ifdef MBP_WITH_CUDA

		Cuda cuda;
		bool useCuda;
		bool CreateCudaNetwork();

		#endif

		/**
		 Attribute : static char * unrecognizedFormat
		 Purpose   : Contains a string specifying that 
		             the file format is not recognized.
		*/
		static const char * unrecognizedFormat;

		/**
		 Attribute : static const int maxPointsRMS
		 Purpose   : Contains the maximum number of 
		             points used to draw the graphics.
		*/
		static const int maxPointsRMS;

		/**
		 Attribute : DWORD programPriority
		 Purpose   : Indicates the priority of the program.
		*/
		DWORD programPriority;

		/**
		 Attribute : bool updateScreen
		 Purpose   : Indicates if the screen is updated during the training.
		*/
		bool updateScreen;

		/**
		 Attribute : CString path
		 Purpose   : Path where the last loaded network was.
		*/
		CString path;

		/**
		 Attribute : VariablesData trainVariables
		 Purpose   : Will contain the train variables data.
		*/
		VariablesData trainVariables;

		/**
		 Attribute : VariablesData testVariables
		 Purpose   : Will contain the test variables data.
		*/
		VariablesData testVariables;

		/**
		 Attribute : Matrix<double> trainInputPatterns
		 Purpose   : Contains the input variables to train the network.
		 Comments  : Columns contain variables and rows the patterns.
		*/
		Matrix<double> trainInputPatterns;

		/**
		 Attribute : Matrix<double> trainDesiredOutputPatterns
		 Purpose   : Desired outputs for the training data.
		 Comments  : Columns contain variables and rows the patterns.
		*/
		Matrix<double> trainDesiredOutputPatterns;

		/**
		 Attribute : Matrix<double> testInputPatterns
		 Purpose   : Contains the input variables to test the network.
		 Comments  : Columns contain variables and rows the patterns.
		*/
		Matrix<double> testInputPatterns;

		/**
		 Attribute : Matrix<double> testDesiredOutputPatterns
		 Purpose   : Desired outputs for the testing data.
		 Comments  : Columns contain variables and rows the patterns.
		*/
		Matrix<double> testDesiredOutputPatterns;

		/**
		 Attribute : Pointer <BackPropagation> mbp
		 Purpose   : Will contain the Multiple Back-Propagation network.
		*/
		Pointer <MultipleBackPropagation> mbp;

		#ifdef MBP_WITH_CUDA
		Pointer <CudaMultipleBackPropagation> mbpCuda;
		#endif

		/**
		 Attribute : long epoch
		 Purpose   : Will contain the actual training epoch.
		*/
		long epoch;

		/**
		 Attribute : Pointer<BPTopologyControl> BPTopologyCtrl
		 Purpose   : Will contain the BPTopology control that allows the user
		             to define the topology of the Back-Propagation network.
		*/
		Pointer<MBPTopologyControl> MBPTopologyCtrl;

		/**
		 Attribute : Pointer<ComboOutput> comboOutput
		 Purpose   : Will contain a pointer to the combo box that allows the
		             user to select the output variable he desires to see.
		*/
		Pointer<ComboOutput> comboOutput;

		/**
		 Attribute : Pointer<CGraphic> trainingOutputGraphic
		 Purpose   : Pointer to the Training Output vs Desired output.
		*/
		Pointer<Graphic> trainingOutputGraphic;

		/**
		 Attribute : Pointer<CGraphic> testingOutputGraphic
		 Purpose   : Pointer to the Testing Output vs Desired output.
		*/
		Pointer<Graphic> testingOutputGraphic;

		/**
		 Attribute : Pointer<CGraphic> RMSGraphic
		 Purpose   : Pointer to the root mean sqare error graphic.
		*/
		Pointer<Graphic> RMSGraphic;

		/**
		 Attribute : Matrix<double> desiredTrainOutputs
		 Purpose   : Desired outputs for the training data.
		 Comments  : Rows contain variables and columns the values,
		*/
		Matrix<double> desiredTrainOutputs;

		/**
		 Attribute : Matrix<double> desiredTestOutputs
		 Purpose   : Desired outputs for the testing data.
		 Comments  : Rows contain variables and columns the values,
		*/
		Matrix<double> desiredTestOutputs;

		/**
		 Attribute : Matrix<double> networkTrainOutputs
		 Purpose   : Network outputs for the training data.
		*/
		Matrix<double> networkTrainOutputs;

		/**
		 Attribute : Matrix<double> networkTestOutputs
		 Purpose   : Network outputs for the testing data.
		*/
		Matrix<double> networkTestOutputs;

		/**
		 Attribute : Array<double> rmsTrain
		 Purpose   : Root mean square error for the training data.
		*/
		Array<double> rmsTrain;

		/**
		 Attribute : Array<double> rmsTest
		 Purpose   : Root mean square error for the testing data.
		*/
		Array<double> rmsTest;

		/**
		 Attribute : Array<double> rmsSpaceTrain
		 Purpose   : Space Network root mean square error for the training data.
		*/
		Array<double> rmsSpaceTrain;

		/**
		 Attribute : Array<double> rmsSpaceTest
		 Purpose   : Space Network root mean square error for the testing data.
		*/
		Array<double> rmsSpaceTest;

		/**
		 Attribute : long rmsInterval
		 Purpose   : Root mean square error graphic drawing interval (in epochs).
		*/
		long rmsInterval;

		/**
		 Attribute : bool reloadTrainingData
		 Purpose   : Indicates if it is necessary to reload the training data.
		*/
		bool reloadTrainingData;

		/**
		 Attribute : CTime lastTimeTrainingDataWasModified
		 Purpose   : Contains the last time the training data was modified.
		*/
		CTime lastTimeTrainingDataWasModified;	

		/**
		 Attribute : bool reloadTestingData
		 Purpose   : Indicates if it is necessary to reload the testing data.
		*/
		bool reloadTestingData;

		/**
		 Attribute : CTime lastTimeTestingDataWasModified
		 Purpose   : Contains the last time the testing data was modified.
		*/
		CTime lastTimeTestingDataWasModified;

		/**
		 Attribute : clock_t trainingTime
		 Purpose   : Contains the total training time.
		*/
		clock_t trainingTime;

		/**
		 Class    : LearningMomentumInformation
		 Puropse  : Represents the information about the learning rate and momentum
		 Author   : Noel de Jesus Mendonça Lopes
		 Date     : 4 of April of 2000
		 Reviewed : Never
		 Version  : 1.0.0 
		*/
		class LearningMomentumInformation {
			private :
				/**
				 Method  : void Copy(LearningMomentumInformation & other)
				 Purpose : Copy the information from another 
				           LearningMomentumInformation object to this object.
				 Version : 1.0.0
				*/
				void Copy(LearningMomentumInformation & other) {
					learningRate = other.learningRate;
					momentum = other.momentum;
				}

			public :
				/**
				 Class    : VariableInformation
				 Puropse  : Represents the information of a varible, 
				            either the learning rate or the momentum.
				 Author   : Noel de Jesus Mendonça Lopes
				 Date     : 4 of April of 2000
				 Reviewed : Never
				 Version  : 1.0.0 
				*/
				class VariableInformation {
					private :
						/**
						 Method  : void Copy(VariableInformation & other)
						 Purpose : Copy the information from another 
						           VariableInformation object to this object.
						 Version : 1.0.0
						*/
						void Copy(VariableInformation & other) {
							value = other.value;
							decayPercentage = other.decayPercentage;
							decayEpochs = other.decayEpochs;
						}

					public :
						/**
						 Attribute : double value
						 Purpose   : Contains the value of the variable.
						*/
						double value;

						/**
						 Attribute : long decayPercentage
						 Purpose   : Contains the decay percentage of the variable.
						*/
						long decayPercentage;

						/**
						 Attribute : long decayEpochs
						 Purpose   : Contains the interval in epochs 
						             where the variable will be decayed.
						*/
						long decayEpochs;

						/**
						 Constructor : VariableInformation()
						 Purpose     : Create a VariableInformation object.
						 Version     : 1.0.0
						*/
						VariableInformation() {}

						/**
						 Constructor : VariableInformation(VariableInformation & other)
						 Purpose     : Create a VariableInformation object from another.
						 Version     : 1.0.0
						*/
						VariableInformation(VariableInformation & other) {
							Copy(other);
						}

						/**
						 Operator : VariableInformation & operator = (VariableInformation & other)
						 Purpose  : Assign the values of other VariableInformation
						            object to this object.
						 Version  : 1.0.0
						*/
						VariableInformation & operator = (VariableInformation & other) {
							Copy(other);
							return *this;
						}
				};

				/**
				 Attribute : VariableInformation learningRate
				 Purpose   : Contains information about the network learning rate.
				*/			
				VariableInformation learningRate;

				/**
				 Attribute : VariableInformation momentum
				 Purpose   : Contains information about the network momentum.
				*/			
				VariableInformation momentum;

				/**
				 Constructor : LearningMomentumInformation()
				 Purpose     : Create a LearningMomentumInformation object.
				 Version     : 1.0.0
				*/
				LearningMomentumInformation() {}

				/**
				 Constructor : LearningMomentumInformation(LearningMomentumInformation & other)
				 Purpose     : Create a LearningMomentumInformation object from another.
				 Version     : 1.0.0
				*/
				LearningMomentumInformation(LearningMomentumInformation & other) {
					Copy(other);
				}

				/**
				 Operator : LearningMomentumInformation & operator = (LearningMomentumInformation & other)
				 Purpose  : Assign the values of other LearningMomentumInformation
				            object to this object.
				 Version  : 1.0.0
				*/
				LearningMomentumInformation & operator = (LearningMomentumInformation & other) {
					Copy(other);
					return *this;
				}
		};

		/**
		 Attribute : BackPropagation::LearningMomentumInformation mainNetLearningMomentumInformation
		 Purpose   : Contains the Learning rate and momentum 
		             information of the main network.
		*/
		LearningMomentumInformation mainNetLearningMomentumInformation;
		
		/**
		 Attribute : BackPropagation::LearningMomentumInformation spaceNetLearningMomentumInformation
		 Purpose   : Contains the Learning rate and momentum 
		             information of the space network.
		*/
		LearningMomentumInformation spaceNetLearningMomentumInformation;

		/**
		 Attribute : bool epochsStop
		 Purpose   : indicates wether the training should stop after a given 
		             number of epochs or after a given training RMS is reached.
		*/
		bool epochsStop;

		/**
		 Attribute : long numberEpochsToStop;
		 Purpose   : If epochsStop is true indicates the number
		             of epochs where the training should stop.
		*/
		long numberEpochsToStop;
	
		/**
		 Attribute : double rmsStop;
		 Purpose   : If epochsStop is false indicates the training RMS 
		             that must be reachead in order to stop the training.
		*/
		double rmsStop;

		/**
		 Attribute : double spaceRmsStop;
		 Purpose   : Specificies the amount of the space RMS error 
		             from which the space network must be trained.
		*/
		double spaceRmsStop;

		/**
		 Attribute : bool autoUpdateLearning
		 Purpose   : indicates if learning rate should be automatically updated.
		*/
		bool autoUpdateLearning;

		/**
		 Attribute : bool deltaBarDelta
		 Purpose   : indicates if the algorithm used to train the 
		             individual networks is the Delta-Bar-Delta.
		*/
		bool deltaBarDelta;

		/**
		 Attribute : double weightDecay
		 Purpose   : Contains the weight decay
		*/
		double weightDecay;

		/**
		 Attribute : bool robustLearning
		 Purpose   : indicates if the we are going to use robust learning.
		*/
		bool robustLearning;

		/**
		 Attribute : double rmsGrowToApplyRobustLearning
		 Purpose   : Percentage of RMS grow that avtivates robust learning.
		*/
		double rmsGrowToApplyRobustLearning;

		/**
		 Attribute : double robustFactor
		 Purpose   : Factor used to decrease the learning 
		             rate when robust learning is used.
		*/
		double robustFactor;

		/**
		 Attribute : bool autoUpdateMomentum
		 Purpose   : indicates if the momentum should be automatically updated.
		*/
		bool autoUpdateMomentum;

		/**
		 Attribute : bool randomizePatterns
		 Purpose   : indicates if the training patterns 
		             should be presented in a random order.
		*/
		bool randomizePatterns;

		/**
		 Attribute : bool batchTraining
		 Purpose   : indicates if the network is training 
		             using the batch or online methods.
		*/
		bool batchTraining;

		/**
		 Method   : void CMBackPropDlg::SelectTopologyTab()
		 Purpose  : Selects the topology tab
		*/
		void SelectTopologyTab();

		/**
		 Method  : void UpdateLearningMomentum()
		 Purpose : Update the values of the learning rate and 
		           momentum for the main and the space network.
		*/
		void UpdateLearningMomentum(); 

		/**
		 Method  : void CreateNetwork()
		 Purpose : Create the network. Returns true if sucessfull 
		           otherwise returns false.
		*/
		bool CreateNetwork();

		/**
		 Method   : bool LoadTrainingData(bool warnUser = true)
		 Purpose  : Load data from the training file, only if it is not loaded yet.
		            Returns true if the training data could be successfuly obtained.
		 Comments : If the training file is allready in the memory this method
		            will determine if the file has change since it was readed. 
								If so the file will be read again.
		*/
		bool LoadTrainingData(bool warnUser = true);

		/**
		 Method   : bool CBackPropDlg::LoadTestingData(bool warnUser = true)
		 Purpose  : Load data from the testing file, only if it is not loaded yet.
		            Returns true if the testing data could be successfuly obtained.
		 Comments : If the testing file is allready in the memory this method
		            will determine if the file has change since it was readed. 
								If so the file will be read again.
		*/
		bool LoadTestingData(bool warnUser = true);

		/**
		Method   : void StopTraining()
		Purpose  : If the network is trainig stop the training.
		*/
		void StopTraining(); 

		/**
		 Method   : bool TopologyIsValid()
		 Purpose  : Returns true if the network topology 
		            is valid and false otherwise.
		 Comments : If the topology is invalid the user is informed 
		            and the focus is set to the BPTopology control.
		*/
		bool TopologyIsValid();

		/**
		 Method   : void EnableOperations(BOOL value)
		 Purpose  : Enable or prevent user from doing operations.
		*/
		void EnableOperations(BOOL value);

		/**
		 Method   : void TrainRMS(double & rms, double & spaceRMS, bool display)
		 Purpose  : Calculates the train root mean square error.
		 Comments : The method also displays the rms value and invalidates 
		            the training output vs desired graphic.
		*/
		void TrainRMS(double & rms, double & spaceRMS, bool display);

		/**
		 Method   : void TestRMS(double & rms, double & spaceRMS)
		 Purpose  : Calculates the test root mean square error.
		 Comments : The method also displays the rms value and invalidates 
		            the testing output vs desired graphic.
		*/
		void TestRMS(double & rms, double & spaceRMS, bool display);

		/**
		 Method   : void LoadTrainingTestDataIfPossible(bool changeTab = true)
		 Purpose  : Load training and test data if possible. Calculate the 
		            corresponding RMS errors and update graphics.
		*/
		void LoadTrainingTestDataIfPossible(bool changeTab = true);

		/**
		 Method   : void FillComboOutput(VariablesData & v)
		 Purpose  : Fullfill the output combo with the network outputs.
		*/
		void FillComboOutput(VariablesData & v);

		/**
		 Attribute : HICON m_hIcon
		 Purpose   : Dialog icon.
		*/
		HICON m_hIcon;

		/**
		 Method   : static UINT TrainNetwork(LPVOID pParam);
		 Purpose  : Train the network.
		*/
		static UINT TrainNetwork(LPVOID pParam);

		enum { IDD = IDD_MBACKPROP_DIALOG };
		CButton	sensitivityButton;
		CStatic	labelSpaceTrainRMSDisplay;
		CStatic	labelSpaceTestRMSDisplay;
		CStatic	testSpaceRMSDisplay;
		CStatic	trainSpaceRMSDisplay;
		CButton	frameMainRMS;
		CButton	frameSpaceRMS;
		CButton	viewWeights;
		CStatic	labelSpaceMomentum;
		CButton	LearningMomentum;
		CButton	frameSpaceLearning;
		CButton	frameMainLearning;
		CStatic	labelSpaceLearning;
		CStatic	labelMainMomentum;
		CStatic	labelMainLearning;
		CButton	frameLearning;
		CButton	saveNetwork;
		CButton	loadNetwork;
		CButton	generateCCode;
		CButton	frameNetwork;
		CButton	saveWeightsButton;
		CButton	randomizeWeightsButton;
		CButton	loadWeightsButton;
		CButton	frameWeights;
		CStatic	labelTrainRMSDisplay;
		CStatic	labelTestRMSDisplay;
		CButton	frameEpoch;
		CButton	frameRMS;
		CButton	frameDataFiles;
		CTabCtrl tabs;
		CButton	stopButton;
		CStatic	trainRMSDisplay;
		CStatic	testRMSDisplay;
		CStatic	epochDisplay;
		CButton	trainButton;
		Filebox m_trainFileBox;
		Filebox m_testFileBox;
		CientificNumber editMainLearning;
		CientificNumber editMainMomentum;
		CientificNumber editSpaceLearning;
		CientificNumber editSpaceMomentum;

		virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV support

		virtual BOOL OnInitDialog();
		afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
		afx_msg void OnPaint();
		afx_msg HCURSOR OnQueryDragIcon();
		afx_msg void OnTrain();
		afx_msg void OnStop();
		afx_msg void OnSelchangeTab(NMHDR* pNMHDR, LRESULT* pResult);
		afx_msg void OnClose();
		afx_msg void OnSize(UINT nType, int cx, int cy);
		afx_msg void OnGetMinMaxInfo(MINMAXINFO FAR* lpMMI);
		afx_msg void OnChangeTestFileBox();
		afx_msg void OnChangeTrainFileBox();
		afx_msg void OnRandomizeWeights();
		afx_msg void OnLoadWeights();
		afx_msg void OnSaveWeights();
		afx_msg void OnGenerateCCode();
		afx_msg void OnLoadNetwork();
		afx_msg void OnSaveNetwork();
		afx_msg void OnLearningMomentum();
		afx_msg void OnViewWeights();
		afx_msg void OnSensivity();
		DECLARE_EVENTSINK_MAP()

		DECLARE_MESSAGE_MAP()

		/**
		 Attribute : volatile bool stopTrain
		 Purpose   : Indicates to the training thread if training should be stoped.
		*/
		volatile bool stopTrain;

		CWinThread * trainingThread;

		/**
		 Method  : void TrainOneEpoch()
		 Purpose : Train one epoch. 
		*/
		void TrainOneEpoch();

		#ifdef MBP_WITH_CUDA
		void TrainOneEpochUsingCuda();
		#endif

		/**
		 Attribute : int numberEpochsBeforeUpdateLearningMomentum
		 Purpose   : Contains the number of epochs before updating 
		             the learning rate and the momentum.
		 Comments  : Used when the learning rate and the
		             momentum are automatically updated.
		*/
//		int numberEpochsBeforeUpdateLearningMomentum;

		/**
		 Attribute : double percentIncDecLearnRate
		 Purpose   : Contains the percentage value used to 
		             increase and decrease the learning rate.
		 Comments  : Used when the learning rate and the
		             momentum are automatically updated.
		*/
		double percentIncDecLearnRate;

		/**
		 Attribute : double percentIncDecSpaceLearnRate
		 Purpose   : Contains the percentage value used to 
		             increase and decrease the space learning rate.
		 Comments  : Used when the learning rate and the
		             momentum are automatically updated.
		*/
		double percentIncDecSpaceLearnRate;

		/**
		 Attribute : double percentIncDecMomentum
		 Purpose   : Contains the percentage value used to 
		             increase and decrease the momentum.
		 Comments  : Used when the learning rate and the
		             momentum are automatically updated.
		*/
		double percentIncDecMomentum;

		/**
		 Attribute : double percentIncDecSpaceMomentum
		 Purpose   : Contains the percentage value used to 
		             increase and decrease the space momentum.
		 Comments  : Used when the learning rate and the
		             momentum are automatically updated.
		*/
		double percentIncDecSpaceMomentum;

		/**
		 Attribute : double previousRMS
		 Purpose   : Contains the previous root mean square error.
		 Comments  : Used when the learning rate and the
		             momentum are automatically updated.
		*/
		//double previousRMS;

		/**
		 Attribute : double previousSpaceRMS
		 Purpose   : Contains the previous root mean square 
		             error for the space network.
		 Comments  : Used when the learning rate and the
		             momentum are automatically updated.
		*/
		//double previousSpaceRMS;

		/**
		 Attribute : double lastRMS
		 Purpose   : Contains the last training RMS. Updated each 100 epochs.
		 Comments  : Used when the learning rate and the
		             momentum are automatically updated.
		*/
//		double lastRMS;

		/**
		 Attribute : double lastSpaceRMS
		 Purpose   : Contains the last training RMS for the space network.
		             Updated each 100 epochs.
		 Comments  : Used when the learning rate and the
		             momentum are automatically updated.
		*/
//		double lastSpaceRMS;

		/**
		 Attribute : double lastRMSVariantion
		 Purpose   : Contains the last training RMS variation.
		 Comments  : Used when the learning rate and the
		             momentum are automatically updated.
		*/
//		double lastRMSVariantion;

		/**
		 Attribute : double lastSpaceRMSVariantion
		 Purpose   : Contains the last space training RMS variation.
		 Comments  : Used when the learning rate and the
		             momentum are automatically updated.
		*/
//		double lastSpaceRMSVariantion;

		/**
		 Attribute : bool increasingLearningRate
		 Purpose   : indicates if the learning rate 
		             should be increased or decreased.
		 Comments  : Used when the learning rate is updated automatically.
		*/
//		bool increasingLearningRate;

		/**
		 Attribute : bool increasingSpaceLearningRate
		 Purpose   : indicates if the space learning rate 
		             should be increased or decreased.
		 Comments  : Used when the learning rate is updated automatically.
		*/
//		bool increasingSpaceLearningRate;

		/**
		 Attribute : bool increasingMomentum
		 Purpose   : indicates if the momentun should be increased or decreased.
		 Comments  : Used when the momentum is updated automatically.
		*/
//		bool increasingMomentum;

		/**
		 Attribute : bool increasingSpaceMomentum
		 Purpose   : indicates if the space momentun 
		             should be increased or decreased.
		 Comments  : Used when the momentum is updated automatically.
		*/
//		bool increasingSpaceMomentum;

		/**
		 Attribute : bool updateLearning
		 Purpose   : indicates whether the main network learning 
		             or momentum should be updated.
		 Comments  : Used when the learning rate and the 
		             momentum are updated automatically.
		*/
//		bool updateLearning;

		/**
		 Attribute : bool updateSpaceLearning
		 Purpose   : indicates whether the space network learning 
		             or momentum should be updated.
		 Comments  : Used when the learning rate and the 
		             momentum are updated automatically.
		*/
//		bool updateSpaceLearning;

		/**
		 Method  : void SetLearningRate(double value)
		 Purpose : Sets the learning rate of the main network and 
		           updates the edit box containing the learning rate.
		 Version : 1.0.0
		*/
		void SetLearningRate(double value) {
			if (autoUpdateLearning) {
				if (value < 0.0000001) {
					value = 0.0000001;
				} else if (value > 1.0) {
					value = 1.0;
				}
			}
			editMainLearning.SetValue(value);
			mbp->SetLearningRate(value);
		}

		/**
		 Method  : void SetSpaceLearningRate(double value)
		 Purpose : Sets the learning rate of the space network and 
		           updates the edit box containing the space learning rate.
		 Version : 1.0.0
		*/
		void SetSpaceLearningRate(double value) {
			if (autoUpdateLearning) {
				if (value < 0.0000001) {
					value = 0.0000001;
				} else if (value > 1.0) {
					value = 1.0;
				}
			}
			editSpaceLearning.SetValue(value);
			mbp->SetSpaceLearningRate(value);
		}

		/**
		 Method  : void SetMomentum(double value)
		 Purpose : Sets the momentum of the main network and 
		           updates the edit box containing the momentum.
		 Version : 1.0.0
		*/
		void SetMomentum(double value) {
			if (value > 0.9 && autoUpdateMomentum) value = 0.9;
			editMainMomentum.SetValue(value);
			mbp->SetMomentum(value);
		}

		/**
		 Method  : void SetSpaceMomentum(double value)
		 Purpose : Sets the momentum of the space network and 
		           updates the edit box containing the space momentum.
		 Version : 1.0.0
		*/
		void SetSpaceMomentum(double value) {
			if (value > 0.9 && autoUpdateMomentum) value = 0.9;
			editSpaceMomentum.SetValue(value);
			mbp->SetSpaceMomentum(value);
		}

		void LoadString(InputFile & f, CString & s);
		int LoadInt(InputFile & f);
		double LoadDouble(InputFile & f);
		bool LoadBool(InputFile & f);
		void LoadLayers(InputFile & f, bool network);

	public:
		/**
		 Constructor : CBackPropDlg()
		 Purpose     : Create a Back-Propagation dialog.
		*/
		CMBackPropDlg();	

		/**
		 Method   : bool GenerateCCode(CString & filename)
		 Purpose  : Generate C code for the network and 
		            save it into the given filename.
		*/
		bool GenerateCCode(CString & filename);

		/**
		 Method   : bool SaveWeights(CString & filename)
		 Purpose  : Save the network weights. Returns 
		            true if succefull, false otherwise. 
		*/
		bool SaveWeights(CString & filename);

		/**
		 Method   : bool LoadWeights(CString & filename)
		 Purpose  : Load the network weights. Returns 
				        true if succefull, false otherwise. 
		*/
		bool LoadWeights(CString & filename);

		/**
		 Method   : bool SaveNetwork(CString & filename)
		 Purpose  : Save the network. Returns true if succefull, false otherwise.
		*/
		bool SaveNetwork(CString & filename);

		/**
		 Method   : bool LoadNetwork(CString & filename)
		 Purpose  : Load the network. Returns true if succefull, false otherwise. 
		*/
		bool LoadNetwork(CString & filename);

		/**
		 Method   : void OutputChanged(int newOutput);
		 Purpose  : Called each time the output variable that the 
		            user is seeing changes. At this point graphics 
								must change in order to reflect the changes.
		*/
		void OutputChanged(int newOutput);

		/**
		 Method  : void ResetNetwork()
		 Purpose : Deletes the Back-Propagation network if it has been created and sets the epoch to 0.
		 Version : 1.0.0
		*/
		void ResetNetwork() {
			mbp = NULL;

			#ifdef MBP_WITH_CUDA
			mbpCuda = NULL;
			#endif

			epoch = 0;
			trainingTime = 0;
			UpdateLearningMomentum();
		}
};