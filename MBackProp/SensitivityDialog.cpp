/*
	Noel Lopes is an Assistant Professor at the Polytechnic Institute of Guarda, Portugal (for more information see readme.txt)
    Copyright (C) 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010 Noel de Jesus Mendonça Lopes

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
 Class    : SensitivityDialog
 Puropse  : Input Sensitivity Dialog class.
 Date     : 13 of September of 2000
 Reviewed : 7 of March of 2009
 Version  : 1.1.0
 Comments : No precompliled headers and /clr and /EAh
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
                            -->| CDialogOld |
                                ---------
                                  |   ------------
                                  -->| GridDialog |
                                      ------------
                                        |   -------------------
                                        -->| SensitivityDialog |
                                           --------------------
*/

#include "stdafx.h"
#include "SensitivityDialog.h"
#include "GridContainer.h"

#define RMS_INCREASE_NEEDED_FOR_TOMATO 0.01
#define RMS_INCREASE_NEEDED_FOR_LAWNGREEN -0.01
#define R_BASE_COLOR 255
#define G_BASE_COLOR 255
#define B_BASE_COLOR 224

/**
 Constructor : void SensitivityDialog::DisplayTitleInGrid(long row, long initialCol, LPCTSTR title)
 Purpose     : Create the Dialog.
 Version     : 1.0.0
*/
SensitivityDialog::SensitivityDialog(CMBackPropDlg * parent) : GridDialog(parent) {}

/**
 Method  : void DisplayTitleInGrid(long row, long col, LPCTSTR title)
 Purpose : Display a title in the grid.
 Version : 1.0.0
*/
void SensitivityDialog::DisplayTitleInGrid(long row, long initialCol, LPCTSTR title) {
	MBPGrid::Grid ^ grid = Container::grid;

	System::String ^ str = gcnew System::String(title);
	if (parent->mbp->HasSpaceNetwork()) {
		grid[row +  2, initialCol] = str;
		grid[row +  15, initialCol] = str;
	} else {
		grid[row, initialCol] = str;
	}
}

/**
 Method  : void SensitivityDialog::SetColor(long row, long column, double RMSincrease)
 Purpose : Set a cell color accordingly to the rms increase.
 Comments: LightYellow (255, 255, 224) | Tomato (255, 99, 71) | LawnGreen (124, 252, 0)
 Version : 1.0.0
*/
void SensitivityDialog::SetColor(long row, long column, double RMSincrease) {
	int R = R_BASE_COLOR;
	int G = G_BASE_COLOR;
	int B = B_BASE_COLOR;

	MBPGrid::Grid ^ grid = Container::grid;

	if (RMSincrease > 0.0) {
		if (RMSincrease > RMS_INCREASE_NEEDED_FOR_TOMATO) RMSincrease = RMS_INCREASE_NEEDED_FOR_TOMATO;

		G -= (int)((G_BASE_COLOR-99) * RMSincrease / RMS_INCREASE_NEEDED_FOR_TOMATO);
		B -= (int)((B_BASE_COLOR-71) * RMSincrease / RMS_INCREASE_NEEDED_FOR_TOMATO);
	} else {
		if (RMSincrease < RMS_INCREASE_NEEDED_FOR_LAWNGREEN) RMSincrease = RMS_INCREASE_NEEDED_FOR_LAWNGREEN;

		R -= (int)((R_BASE_COLOR-124) * RMSincrease / RMS_INCREASE_NEEDED_FOR_LAWNGREEN);
		B -= (int)((G_BASE_COLOR-252) * RMSincrease / RMS_INCREASE_NEEDED_FOR_LAWNGREEN);
		B -= (int)((B_BASE_COLOR-0) * RMSincrease / RMS_INCREASE_NEEDED_FOR_LAWNGREEN);
	}

	grid->SetCellColor(row, column, R, G, B);
}

/**
 Method  : BOOL SensitivityDialog::OnInitDialog()
 Purpose : Calculate and display the sensitivity of the RMS to input noise.
 Version : 1.0.0
*/
BOOL SensitivityDialog::OnInitDialog() {
	SetCursor(AfxGetApp()->LoadStandardCursor(IDC_WAIT));

	CDialog::OnInitDialog();

	CString s;
	int colTestVars;

	SetWindowText(L"Network sensitivity to inputs");

	MultipleBackPropagation * mbp = parent->mbp;

	int inputs = mbp->Inputs();

	int numberVars = inputs + mbp->Outputs();

	int numberTrainVariables = parent->trainVariables.Number();
	int numberTestVariables = parent->testVariables.Number();

	int lastCol = (numberTrainVariables == numberVars && numberTestVariables == numberVars) ? inputs << 1 : inputs;

	MBPGrid::Grid ^ grid = Container::grid;

	grid->Rows = (mbp->HasSpaceNetwork()) ? 26 : 11;
	grid->Columns = lastCol + 1;

	if (mbp->HasSpaceNetwork()) {
		grid[1,1] = gcnew System::String(L"/PMain Network");
		grid[14,1] = gcnew System::String(L"/PSpace Network");
	}

	DisplayTitleInGrid(1, 1, L"/TRMS after the addition of noise to the");

	DisplayTitleInGrid(3, 0, L"/TNoise");
	DisplayTitleInGrid(4, 0, L"/S0%");
	DisplayTitleInGrid(5, 0, L"/S2%");
	DisplayTitleInGrid(6, 0, L"/S5%");
	DisplayTitleInGrid(7, 0, L"/S10%");
	DisplayTitleInGrid(8, 0, L"/Salways set to minimum");
	DisplayTitleInGrid(9, 0, L"/Salways set to mean");
	DisplayTitleInGrid(10, 0, L"/Salways set to maximum");

	if (lastCol > inputs) {
		DisplayTitleInGrid(2, 1, L"/TTraining Data");
		DisplayTitleInGrid(2, inputs + 1, L"/TTesting Data");

		for (int i = 1; i <= inputs; i++) {
			s.Format(L"/S%dth input", i);
			DisplayTitleInGrid(3, i, s);
			DisplayTitleInGrid(3, inputs + i, s);
		}
	} else {
		DisplayTitleInGrid(2, 1, (numberTrainVariables == numberVars) ? L"/TTraining Data" : L"/TTesting Data");

		for (int i = 1; i <= inputs; i++) {
			s.Format(L"/S%dth input", i);
			DisplayTitleInGrid(3, i, s);
		}
	}

	double rms, spaceRMS;
	double initialRMS, initialSpaceRMS;

	int mainNetCol = (mbp->HasSpaceNetwork()) ? 6 : 4;
	
	if (numberTrainVariables == numberVars) {
		mbp->CalculateRMS(initialRMS, initialSpaceRMS, parent->trainInputPatterns, parent->trainDesiredOutputPatterns, NULL);
		
		s.Format(L"%1.10f", initialRMS);
		System::String ^ str = gcnew System::String(s);
		for (int i = 1; i <= inputs; i++) {
			grid[mainNetCol, i] = str;
			grid->SetCellColor(mainNetCol, i, R_BASE_COLOR, G_BASE_COLOR, B_BASE_COLOR);
		}

		if (mbp->HasSpaceNetwork()) {
			s.Format(L"%1.10f", initialSpaceRMS);
			str = gcnew System::String(s);
			for (int i = 1; i <= inputs; i++) {
				grid[19, i] = str;
				grid->SetCellColor(19, i, R_BASE_COLOR, G_BASE_COLOR, B_BASE_COLOR);
			}
		}

		Matrix<double> trainInputPatterns(parent->trainInputPatterns.Rows() << 1, parent->trainInputPatterns.Columns());
		memcpy(trainInputPatterns.Pointer(), parent->trainInputPatterns.Pointer(), parent->trainInputPatterns.Elements() * sizeof(double));
		memcpy((trainInputPatterns.Pointer() + parent->trainInputPatterns.Elements()), parent->trainInputPatterns.Pointer(), parent->trainInputPatterns.Elements() * sizeof(double));

		Matrix<double> trainDesiredOutputPatterns(parent->trainDesiredOutputPatterns.Rows() << 1, parent->trainDesiredOutputPatterns.Columns());
		memcpy(trainDesiredOutputPatterns.Pointer(), parent->trainDesiredOutputPatterns.Pointer(), parent->trainDesiredOutputPatterns.Elements() * sizeof(double));
		memcpy((trainDesiredOutputPatterns.Pointer() + parent->trainDesiredOutputPatterns.Elements()), parent->trainDesiredOutputPatterns.Pointer(), parent->trainDesiredOutputPatterns.Elements() * sizeof(double));

		int rows = parent->trainInputPatterns.Rows();

		for (int i = 0; i < inputs; i++) {
			for (int n = 1; n < 7; n++) {
				if (n < 4) {
					double noise = (n + 1) * 0.02;
					for (int r = 0; r < rows; r++) {
						if (trainInputPatterns[r][i] != MISSING_VALUE) {
							trainInputPatterns[r][i] += noise;
							trainInputPatterns[rows + r][i] -= noise;
						}
					}
				} else {
					double noise = -1.0 + (n - 4) * +1.0;
					for (int r = 0; r < (rows << 1); r++) trainInputPatterns[r][i] = noise;
				}

				mbp->CalculateRMS(rms, spaceRMS, trainInputPatterns, trainDesiredOutputPatterns, NULL);

				CString s;
				s.Format(L"%1.10f", rms);
				grid[mainNetCol + n, i + 1] = gcnew System::String(s);
				SetColor(mainNetCol + n, i + 1, rms - initialRMS);

				if (mbp->HasSpaceNetwork()) {
					CString s;
					s.Format(L"%1.10f", spaceRMS);
					grid[19 + n, i + 1] = gcnew System::String(s);
					SetColor(19 + n, i + 1, spaceRMS - initialSpaceRMS);
				}
			}

			if (i != inputs - 1) {
				for (int r = 0; r < rows; r++) trainInputPatterns[rows + r][i] = trainInputPatterns[r][i] = parent->trainInputPatterns[r][i];
			}
		}

		colTestVars = inputs;
	} else {
		colTestVars = 0;
	}

	if (numberTestVariables == numberVars) {
		mbp->CalculateRMS(initialRMS, initialSpaceRMS, parent->testInputPatterns, parent->testDesiredOutputPatterns, NULL);

		s.Format(L"%1.10f", initialRMS);
		System::String ^ str = gcnew System::String(s);
		for (int i = 1; i <= inputs; i++) {
			grid[mainNetCol, colTestVars + i] = str;
			grid->SetCellColor(mainNetCol, colTestVars + i, R_BASE_COLOR, G_BASE_COLOR, B_BASE_COLOR);
		}

		if (mbp->HasSpaceNetwork()) {
			s.Format(L"%1.10f", initialSpaceRMS);
			str = gcnew System::String(s);
			for (int i = 1; i <= inputs; i++) {
				grid[19, colTestVars + i] = str;
				grid->SetCellColor(19, colTestVars + i, R_BASE_COLOR, G_BASE_COLOR, B_BASE_COLOR);
			}
		}

		Matrix<double> testInputPatterns(parent->testInputPatterns.Rows() << 1, parent->testInputPatterns.Columns());
		memcpy(testInputPatterns.Pointer(), parent->testInputPatterns.Pointer(), parent->testInputPatterns.Elements() * sizeof(double));
		memcpy((testInputPatterns.Pointer() + parent->testInputPatterns.Elements()), parent->testInputPatterns.Pointer(), parent->testInputPatterns.Elements() * sizeof(double));

		Matrix<double> testDesiredOutputPatterns(parent->testDesiredOutputPatterns.Rows() << 1, parent->testDesiredOutputPatterns.Columns());
		memcpy(testDesiredOutputPatterns.Pointer(), parent->testDesiredOutputPatterns.Pointer(), parent->testDesiredOutputPatterns.Elements() * sizeof(double));
		memcpy((testDesiredOutputPatterns.Pointer() + parent->testDesiredOutputPatterns.Elements()), parent->testDesiredOutputPatterns.Pointer(), parent->testDesiredOutputPatterns.Elements() * sizeof(double));

		int rows = parent->testInputPatterns.Rows();

		for (int i = 0; i < inputs; i++) {
			for (int n = 1; n < 7; n++) {
				if (n < 4) {
					double noise = (n + 1) * 0.02;
					for (int r = 0; r < rows; r++) {
						if (testInputPatterns[r][i] != MISSING_VALUE) {
							testInputPatterns[r][i] += noise;
							testInputPatterns[rows + r][i] -= noise;
						}
					}
				} else {
					double noise = -1.0 + (n - 4) * 1.0;
					for (int r = 0; r < (rows << 1); r++) testInputPatterns[r][i] = noise;
				}

				mbp->CalculateRMS(rms, spaceRMS, testInputPatterns, testDesiredOutputPatterns, NULL);

				CString s;
				s.Format(L"%1.10f", rms);
				grid[mainNetCol + n, colTestVars + i + 1] = gcnew System::String(s);
				SetColor(mainNetCol + n, colTestVars + i + 1, rms - initialRMS);

				if (mbp->HasSpaceNetwork()) {
					CString s;
					s.Format(L"%1.10f", spaceRMS);
					grid[19 + n, colTestVars + i + 1] = gcnew System::String(s);
					SetColor(19 + n, colTestVars + i + 1, spaceRMS - initialSpaceRMS);
				}
			}

			if (i != inputs - 1) {
				for (int r = 0; r < rows; r++) testInputPatterns[rows + r][i] = testInputPatterns[r][i] = parent->testInputPatterns[r][i];
			}
		}
	}

	SetCursor(AfxGetApp()->LoadStandardCursor(IDC_ARROW));

	return TRUE;
}