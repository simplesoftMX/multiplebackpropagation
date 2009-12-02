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
#ifndef LearningMomentumDialog_h
#define LearningMomentumDialog_h

#include "MBackPropDlg.h"
#include "CtrlUnsignedEdit.h"
#include "CtrlCientificNumber.h"
#include "afxwin.h"

#if _MSC_VER > 1000
	#pragma once
#endif

class LearningMomentumDialog : public CDialog {
	private :
		//{{AFX_DATA(LearningMomentumDialog)
		enum { IDD = IDD_LEARNING_MOMENTUM };

		CComboBox	comboThread;
		CButton	updateScreen;
		CStatic	lbSpaceMomentum;
		CStatic	lbSpaceLearning;
		CStatic	lbMainMomentum;
		CStatic	lbMainLearning;
		CButton	useRobustness;
		CButton	deltaBarDeltaButton;
		CButton	autoUpdateMomentumButton;
		CButton	autoUpdateLearningButton;
		CButton RandomizePatterns;
		CButton cbCUDA;
		CButton radioBatch;
		CButton radioOnline;
		UnsignedEdit	mainLearningEpochDecay;
		UnsignedEdit	mainLearningPercentageDecay;
		CientificNumber	mainLearning;
		UnsignedEdit	mainMomentumEpochDecay;
		UnsignedEdit	mainMomentumPercentageDecay;
		CientificNumber	mainMomentum;
		CientificNumber	spaceLearning;
		UnsignedEdit	spaceLearningEpochDecay;
		UnsignedEdit	spaceLearningPercentageDecay;
		CientificNumber	spaceMomentum;
		UnsignedEdit	spaceMomentumEpochDecay;
		UnsignedEdit	spaceMomentumPercentageDecay;
		UnsignedEdit	epochsToStop;
		CientificNumber rmsToStop;
		CientificNumber	d;
		CientificNumber	u;
		CientificNumber	robustnessReduceFactor;
		CientificNumber	applyRobustnessWhenRMSGrows;
		CientificNumber	weightDecay;
		CientificNumber	spaceRmsToStop;
		//}}AFX_DATA

		/**
		 Attribute : CBackPropDlg * parent
		 Purpose   : Contains a pointer to the Multiple 
		             Back-Propagation dialog window.
		*/
		CMBackPropDlg * parent;

		#ifdef MBP_WITH_CUDA
			bool hasConnectionsBetweenInputsAndOutputs;
			bool invalidNetworkTypeCuda;
			bool invalidActivationFunctionsCuda;

			bool makeChangesIfNeededForCuda;

			void CheckForInvalidLayersCuda(bool network);
		#endif
		
	public:
		//{{AFX_VIRTUAL(LearningMomentumDialog)
		virtual void DoDataExchange(CDataExchange* pDX);
		//}}AFX_VIRTUAL

		//{{AFX_MSG(LearningMomentumDialog)
		virtual void OnOK();
		virtual BOOL OnInitDialog();
		afx_msg void OnAutoUpdateChange();
		afx_msg void OnChangeRmsToStop();
		afx_msg void OnChangeRMSToStop();
		afx_msg void OnChangeEpochsToStop();
		afx_msg void OnBatch();
		afx_msg void OnOnline();
		afx_msg void OnAutoUpdateLearning();
		afx_msg void OnAutoUpdateMomentum();
		afx_msg void OnDeltaBarDeltaChange();
		afx_msg void OnRobustness();
		afx_msg void OnStopRms();
		afx_msg void OnStopEpochs();
		afx_msg void OnBnClickedEnableCuda();
		DECLARE_EVENTSINK_MAP()
		//}}AFX_MSG

		DECLARE_MESSAGE_MAP()

	public:
		/**
		 Constructor : LearningMomentumDialog(CBackPropDlg * parent)
		 Purpose     : Initialize the dialog.
		 Version     : 1.0.0
		*/
		LearningMomentumDialog(CMBackPropDlg * parent) : CDialog(IDD_LEARNING_MOMENTUM, parent) {
			this->parent = parent;
		}
};

//{{AFX_INSERT_LOCATION}}

#endif
