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
 Class    : BPTopologyWnd
 Puropse  : Back Propagation Topology window.
 Date     : 3 of February of 2000
 Reviewed : 29 of November of 2009
 Version  : 1.0.2
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
                            |   -----
                            -->| Wnd |
                                -----
                                  |   ---------------
                                  -->| BPTopologyWnd |
                                      ---------------
*/
#include "stdafx.h"
#include "BPTopologyWnd.h"
#include "MBPTopologyCtrl.h"
#include "MBPTopology.h"
#include "../Common/Rect/Rect.h"

#ifdef _DEBUG
	#define new DEBUG_NEW
	#undef THIS_FILE
	static char THIS_FILE[] = __FILE__;
#endif


BEGIN_MESSAGE_MAP(BPTopologyWnd, CWnd)
	//{{AFX_MSG_MAP(BPTopologyWnd)
	ON_WM_CREATE()
	ON_WM_SIZE()
	ON_WM_PAINT()
	ON_WM_LBUTTONDOWN()
	ON_WM_CTLCOLOR()
	//}}AFX_MSG_MAP
END_MESSAGE_MAP()

/**
 Method  : short BPTopologyWnd::GetActivationFunction(long layer, long neuron)
 Purpose : Return the activation function of a given 
           neuron in a given layer.
 Version : 1.0.0
*/
short BPTopologyWnd::GetActivationFunction(long layer, long neuron) {
	if (layer >= layersInfo.Lenght()) return -1;

	Array<NeuronInfo> * neuronInfo = &(layersInfo.Element(layer)->neuronInfo);
	
	if (neuron >=  neuronInfo->Lenght()) return -1;

	return (*neuronInfo)[neuron].activationFunction;
}

/**
 Method  : double BPTopologyWnd::GetActivationFunctionParameter(long layer, long neuron)
 Purpose : Return the activation function parameter
           of a given neuron in a given layer.
 Version : 1.0.0
*/
double BPTopologyWnd::GetActivationFunctionParameter(long layer, long neuron) {
	if (layer >= layersInfo.Lenght()) return 0.0;

	Array<NeuronInfo> * neuronInfo = &(layersInfo.Element(layer)->neuronInfo);
	
	if (neuron >=  neuronInfo->Lenght()) return 0.0;

	return (*neuronInfo)[neuron].k;
}

/**
 Method  : int BPTopologyWnd::OnCreate(LPCREATESTRUCT lpCreateStruct)
 Purpose : Create the objects that compose the BPTopology window.
 Version : 1.0.0
*/
int BPTopologyWnd::OnCreate(LPCREATESTRUCT lpCreateStruct) {
	if (CWnd::OnCreate(lpCreateStruct) == -1) return -1;

	CRect ctrlRect;
	GetClientRect(ctrlRect);

	// Create the combo box for selecting the neurons activation function
	comboActivationFunction = new ComboLayerActivationFunction();
	if (comboActivationFunction.IsNull()) return -1;
	if (!comboActivationFunction->Create(ctrlRect, this, false)) return -1;	

	if (comboActivationFunction->AddString(_TEXT("Sigmoid")) < 0) return -1;
	if (comboActivationFunction->AddString(_TEXT("Tanh")) < 0) return -1;
	if (comboActivationFunction->AddString(_TEXT("Gaussian")) < 0) return -1;
	if (comboActivationFunction->AddString(_TEXT("Linear")) < 0) return -1;

	// Create the edition control for editing the k parameter
	parameterTextBox = new EditActivationFunctionParameter();
	if (parameterTextBox.IsNull()) return -1;
	if (!parameterTextBox->Create(ctrlRect, this, false)) return -1;

	// Create the edition control for editing the layers neurons
	textBox = new EditNeuronsLayer();
	if (textBox.IsNull()) return -1;
	if (!textBox->Create(ctrlRect.left, ctrlRect.top, ctrlRect.right, ctrlRect.top + 21, this)) return -1;

	// Create a check box for the connections between the input and the output layers
	if (!inputToOutputConnections.Create(_TEXT("Add connections between the input and the output layers"), ctrlRect.left, ctrlRect.top, ctrlRect.right, ctrlRect.bottom, this, BS_FLAT)) return -1;
	inputToOutputConnections.EnableWindow(FALSE);

	if (spaceNetwork != NULL) {
		editNumberNeuronsWithSelectiveActivation = new EditUnsignedInt();
		if (editNumberNeuronsWithSelectiveActivation.IsNull()) return -1;
		if (!editNumberNeuronsWithSelectiveActivation->Create(ctrlRect, this, false)) return -1;
	}

	return 0;
}

/**
 Method   : void BPTopologyWnd::OnPaint()
 Purpose  : Draw the Back-Propagation network.
 Version  : 1.1.1
*/
void BPTopologyWnd::OnPaint() {
	FlickerFreeDC dc(this);

	dc.SelectFont(textBox->GetFont());

	CRect clientRect;
	GetClientRect(clientRect);

	if (selectedLayer != -1) {
		CRect r;

		CString layerText;
		if (selectedLayer == layersInfo.Lenght()-1) {
			layerText = _TEXT(" Output layer activation function ");
		} else {
			layerText.Format(_TEXT(" Hidden layer #%d activation function "), selectedLayer);
		}			

		CSize textExtent = dc.GetTextExtent(layerText);
		dc.TextOut(0, (textBoxHeight - textExtent.cy) / 2, layerText);

		CRect comboRect;
		comboActivationFunction->GetWindowRect(comboRect);
		ScreenToClient(comboRect);
		dc.TextOut(comboRect.right, (textBoxHeight - textExtent.cy) / 2, _TEXT(" k"));

		if (selectedNeuron != -1) {
			CString neuronNumber;
			neuronNumber.Format(_TEXT("%dth"), selectedNeuron + 1);

			textExtent = dc.GetTextExtent(neuronNumber);
			int neuronSize = max(textExtent.cx, textExtent.cy) << 1;

			r.left   = clientRect.left + neuronSize / 2;
			r.right  = r.left + neuronSize;
			r.top    = (clientRect.Height() - neuronSize) / 2;
			r.bottom = r.top + neuronSize;
			dc.Ellipse(r);

			dc.TextOut(r.left + (r.Width() - textExtent.cx) / 2, r.top + (r.Height() - textExtent.cy) / 2, neuronNumber);
		} else {
			if (spaceNetwork != NULL) {
				CMBPTopologyCtrl * BPTopologyCtrl = static_cast<CMBPTopologyCtrl *>(GetParent());
				if (BPTopologyCtrl->networkType == MBPHO) dc.TextOut(0, clientRect.bottom - textExtent.cy - (textBoxHeight - textExtent.cy) / 2, _TEXT(" Number of neurons with selective activation"));
			}

			LayerInfo * l = layersInfo.Element(selectedLayer);
		
			int neurons = l->neuronInfo.Lenght();

			int neuronSize = clientRect.Height() / (neurons + 1 + (neurons  + 2) / 2);
			if (!neuronSize) return;

			r.left = clientRect.left + neuronSize / 2;
			r.right = r.left + neuronSize;

			int heightBetweenNeurons = (clientRect.Height() - neuronSize * neurons) / (neurons + 1);
			int freeHeight = clientRect.Height() - textBoxHeight - (heightBetweenNeurons * (neurons + 1) + neuronSize * neurons);

			r.top = clientRect.top + textBoxHeight + freeHeight / 2 + heightBetweenNeurons;

			for (int n = 0; n < neurons; n++) {
				r.bottom = r.top + neuronSize;
				dc.Ellipse(r);
				l->neuronInfo[n].position = r;
				r.top = r.bottom + heightBetweenNeurons;
			}
		}

		// Draw the activation function graphic
		int bitmapID = 0;
		switch (comboActivationFunction->GetCurSel()) {
			case Sigmoid :
				bitmapID = IDB_SIGMOID;
				break;
			case Tanh :
				bitmapID = IDB_TANH;
				break;
			case Gaussian :
				bitmapID = IDB_GAUSSIAN;
				break;
			case Linear :
				bitmapID = IDB_LINEAR;
		}

		if (bitmapID) {
			Bitmap activationFunctionBitmap;
			activationFunctionBitmap.LoadBitmap(bitmapID);

			r.left = r.right + 3;
			r.right = clientRect.right - 3;
			r.top = clientRect.top + textBoxHeight + 3;
			r.bottom = clientRect.bottom -3;

			int width = r.Width();
			int height = (int) (width / 1.3333333);
			if (r.Height() < height) {
				height = r.Height();
				width = (int) (height * 1.3333333);
			}	
		
			if (width > activationFunctionBitmap.Width() || height > activationFunctionBitmap.Height()) {
				width = activationFunctionBitmap.Width();
				height = activationFunctionBitmap.Height();
			}
		
			CBitmap * activationFunctionShrinkedBitmap = activationFunctionBitmap.ShrinkBitmap(width, height);
			dc.DrawBitmap(activationFunctionShrinkedBitmap, r.left + (r.Width() - width)/2, r.top + (r.Height() - height)/2);
		}
		
		// Draw the back arrow
		dc.DrawBitmap(IDB_BACK, clientRect.right - 32, clientRect.bottom - 32, SRCAND);
	} else {
		CString text;
		if (textBox.IsNull()) return;			
		
		textBox->GetWindowText(text);

		CRect drawingArea = clientRect;
		drawingArea.top = textBoxHeight;
		drawingArea.bottom -= textBoxHeight;

		DrawNetwork(dc, drawingArea, UpdateInfo(text));
	}
}

/**
 Method   : int BPTopologyWnd::DrawNetwork(FlickerFreeDC & dc)
 Purpose  : Draw the Back-Propagation network. Returns the neuron size.
 Version  : 1.0.0
 Comments : Drawing may not take place in this window.
 
            h = Neuron height
            H = Drawing area height
            w = Neuron width
            W = Drawing area width
            n = Number neurons

            One extra neuron is considerated

             -----------------
            |  h/2            |
	        |  --             |
	        | |  | h          |
            |  --             |
            |  h/2            | H
            |  --             |
            | |  | h          |
            |  --             |
            |  h/2            |
             -----------------

            H = (n+1)h + ((n+2)/2)h <=>
            h = H / (n+1+(n+2)/2)

             --------------------
            |                    |
            |     --      --     |
	        | 3w !  ! 3w !  ! 3w |
	        |     --      --     |
            |      w       w     |
            |                    |
            |                    |
            |                    |
            |                    |
             --------------------
                      W

            W = (n+1)w + ((n+2)*3)w <=>
            w = W / (n+1+(n+2)*3)
*/
int BPTopologyWnd::DrawNetwork(FlickerFreeDC & dc, const CRect & drawingArea, int maxNumberNeuronsPerLayer, bool includeLastLayer) {
	int layers = layersInfo.Lenght();

	if (!layers) {
		RECT r((RECT)drawingArea);
	
		r.left += 21;
		r.right -= 21;
		r.top += 7;
		r.bottom -= 21;

		CFont font;
		font.CreatePointFont(120, L"Times new roman");
		dc.SelectFont(&font);

		CString license = L"Copyright (C) 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010 Noel de Jesus Mendonça Lopes\n";
		license += "This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version. ";
		license += "This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details. ";
		license += "You should have received a copy of the GNU General Public License along with this program.  If not, see <http://www.gnu.org/licenses/>.\n";

		if (mainNetwork == NULL) {
			CString s = L"Please properly cite our work if you find it useful. This supports future development. The following articles provided the foundation for Multiple Back-Propagation software (A bib file <mbp.bib> is included with this software):";

			r.top += 10 + dc.DrawText(s, s.GetLength(), &r, DT_WORDBREAK | DT_NOPREFIX);
			r.left += 7;

			s = L"Lopes, N. and Ribeiro, B. (2010). A Strategy for Dealing With Missing Values by Using Selective Activation Neurons in a Multi-Topology Framework. IEEE World congress on Computational Intelligence WCCI 2010.";
			r.top += 10 + dc.DrawText(s, s.GetLength(), &r, DT_WORDBREAK | DT_NOPREFIX);

			s = L"Lopes, N. and Ribeiro, B. (2009). GPU Implementation of the Multiple Back-Propagation Algorithm, In Proceedings of Intelligent Data Engineering and Automated Learning. Lecture Notes in Computer Science, Springer Berlin / Heidelberg, volume 5788, pages 449-456.";
			r.top += 10 + dc.DrawText(s, s.GetLength(), &r, DT_WORDBREAK | DT_NOPREFIX);

			s = L"Lopes, N. and Ribeiro, B. (2003). An Efficient Gradient-Based Learning Algorithm Applied to Neural Networks with Selective Actuation Neurons. In Neural, Parallel & Scientific Computations, volume 11, pages 253-272. Dynamic Publishers.";
			r.top += 10 + dc.DrawText(s, s.GetLength(), &r, DT_WORDBREAK | DT_NOPREFIX);

			s = L"Lopes, N. and Ribeiro, B. (2001). Hybrid learning in a multi neural network architecture. In INNS-IEEE International Joint Conference on Neural Networks, IJCNN'01, volume 4, pages 2788-2793, Washington D.C., USA.\n";
			r.top += dc.DrawText(s, s.GetLength(), &r, DT_WORDBREAK | DT_NOPREFIX);

			r.left -= 7;

			dc.Line(r.left, r.top, r.right, r.top);
			r.top += 7;
			
			if (!spaceNetwork->IsWindowVisible()) {
				license = L"\n" + license;
				r.top += 7 + dc.DrawText(license, license.GetLength(), &r, DT_WORDBREAK | DT_NOPREFIX);
				dc.Line(r.left, r.top, r.right, r.top);
				r.top += 7;
			}
			
			s = L"\nThis program can be freely obtained on the site http://dit.ipg.pt/MBP";
			dc.DrawText(s, s.GetLength(), &r, DT_WORDBREAK | DT_NOPREFIX);
		} else {
			r.top += 7 + dc.DrawText(license, license.GetLength(), &r, DT_WORDBREAK | DT_NOPREFIX);
		}

		return 0;
	}

	if (!maxNumberNeuronsPerLayer) {
		for(LayerInfo * l = layersInfo.First(); l != NULL; l = layersInfo.Next()) {
			int neurons = l->neuronInfo.Lenght();

			if (neurons > maxNumberNeuronsPerLayer) maxNumberNeuronsPerLayer = neurons;
		}
	}

	// Determine the neurons size
	//int drawingArea.Height() = drawingArea.Height() - textBoxHeight;
	int neuronWidth = drawingArea.Width() / (layers + 1 + (layers + 2) * 3);
	int neuronHeight = drawingArea.Height() / (maxNumberNeuronsPerLayer + 1 + (maxNumberNeuronsPerLayer + 2) / 2);
	int neuronSize = min(neuronWidth, neuronHeight);
	if (!neuronSize) return 0;

	int widthBetweenNeurons = (drawingArea.Width() - neuronSize * layers) / (layers + 1);
	int freeWidth = drawingArea.Width() - (widthBetweenNeurons * (layers + 1) + neuronSize * layers);

	CRect r;
	r.left = drawingArea.left + freeWidth / 2 + widthBetweenNeurons;

	bool drawInputToOutputConnections = (layers > 2 && inputToOutputConnections.GetCheck() && includeLastLayer);

	int layerNumber = 1;
	LayerInfo * previousLayer = NULL;
	LayerInfo * firstLayer = layersInfo.First();

	for(LayerInfo * l = firstLayer; l != NULL; l = layersInfo.Next()) {
		r.right = r.left + neuronSize;

		// Determine how many neurons this layer has.
		int neurons = l->neuronInfo.Lenght();
				
		int heightBetweenNeurons = (drawingArea.Height() - neuronSize * neurons) / (neurons + 1);
		int freeHeight = drawingArea.Height() - (heightBetweenNeurons * (neurons + 1) + neuronSize * neurons);

		Rect biasRect;
		if (layerNumber > 1) {
			biasRect.left   = r.left - widthBetweenNeurons * 2/3 - neuronSize / 4; // the bias is a half size of a neuron
			biasRect.right  = biasRect.left + neuronSize / 2;
			biasRect.bottom = drawingArea.bottom - heightBetweenNeurons / 2;
			biasRect.top    = biasRect.bottom - neuronSize / 2;
		}

		r.top = drawingArea.top + freeHeight / 2 + heightBetweenNeurons;

		for (int n = 0; n < neurons; n++) {
			r.bottom = r.top + neuronSize;

			int y  = r.top + r.Height() / 2;

			if (layerNumber > 1) {
				Array<NeuronInfo> * previousLayerNeuronInfo = &previousLayer->neuronInfo;

				int neuronsPreviousLayer = previousLayerNeuronInfo->Lenght();
				for (int pln = 0; pln < neuronsPreviousLayer; pln++) {
					CRect neuronRect = (*previousLayerNeuronInfo)[pln].position;
					if (layerNumber != layers || includeLastLayer) dc.Line(neuronRect.CenterPoint(), r.CenterPoint());
					//if (layerNumber == 2) {
					//	dc.Rectangle(neuronRect);
					//} else {
					//	dc.Ellipse(neuronRect);
					//}				
				}

				if (layerNumber != layers || includeLastLayer) dc.Line(biasRect.CenterPoint(), r.CenterPoint());
			} else {
				dc.HorizontalArrow(y, r.left - 1 - widthBetweenNeurons / 3, r.left - 1);
			}

			if (layerNumber == layers && includeLastLayer) {
				if (drawInputToOutputConnections) {
					Array<NeuronInfo> * inputLayerNeuronInfo = &firstLayer->neuronInfo;

					int neuronsInputLayer = inputLayerNeuronInfo->Lenght();

					for (int in = 0; in < neuronsInputLayer; in++) {
						CRect neuronRect = (*inputLayerNeuronInfo)[in].position;
						dc.Line(neuronRect.CenterPoint(), r.CenterPoint());
					}
				}

				//if (layerNumber == 1) {
				//	dc.Rectangle(r);
				//} else {
				//	dc.Ellipse(r);
				//}

				dc.HorizontalArrow(y, r.right, r.right + widthBetweenNeurons / 3);
			}

			l->neuronInfo[n].position = r;

			r.top = r.bottom + heightBetweenNeurons;
		}

		if (layerNumber > 1 && (layerNumber != layers || includeLastLayer)) dc.Triangle(CPoint(biasRect.left + biasRect.Width() / 2, biasRect.top), biasRect.BottomLeft(), biasRect.BottomRight());

		r.left = r.right + widthBetweenNeurons;

		previousLayer = l;
		layerNumber++;		
	}

	/*if (drawInputToOutputConnections) {
		Array<NeuronInfo> * inputLayerNeuronInfo = &firstLayer->neuronInfo;

		int neuronsInputLayer = inputLayerNeuronInfo->Lenght();

		for (int in = 0; in < neuronsInputLayer; in++) {
			CRect neuronRect = (*inputLayerNeuronInfo)[in].position;
			dc.Rectangle(neuronRect);
		}


		LayerInfo * lastLayer = layersInfo.Last();

		Array<NeuronInfo> * outputLayerNeuronInfo = &lastLayer->neuronInfo;

		int neuronsOutputLayer = outputLayerNeuronInfo->Lenght();

		for (int o = 0; o < neuronsOutputLayer; o++) {
			CRect neuronRect = (*outputLayerNeuronInfo)[o].position;
			dc.Ellipse(neuronRect);
		}
	}*/

	//for(LayerInfo * l = layersInfo.First(); l != NULL; l = layersInfo.Next()) {

	bool rectangle = true;	
	LayerInfo * nextLayer = NULL;

	for(LayerInfo * l = layersInfo.First(); l != NULL; l = nextLayer) {
		nextLayer = layersInfo.Next();
		if (nextLayer == NULL && !includeLastLayer) break;

		Array<NeuronInfo> * lInfo = &(l->neuronInfo);
		
		int neurons = lInfo->Lenght();

		for (int n = 0; n < neurons; n++) {
			CRect neuronRect = (*lInfo)[n].position;

			if(rectangle) {
				dc.Rectangle(neuronRect);
			} else {
				dc.Ellipse(neuronRect);
			}
		}

		rectangle = false;
	} 

	return neuronSize;
}

/**
 Method  : void BPTopologyWnd::OnFontChanged()
 Purpose : Change the the text boxes and combo boxes fonts
           and adjust the text boxes height.
 Version : 1.0.0
*/
void BPTopologyWnd::FontChanged(CFont * font) {
	TEXTMETRIC textMetrics;
	
	CDC * dc = GetDC();
	CFont * oldFont = dc->SelectObject(font);

	if (dc->GetTextMetrics(&textMetrics)) {
		RECT ctrlRect;
		GetClientRect(&ctrlRect);
		textBoxHeight = textMetrics.tmHeight + textMetrics.tmInternalLeading + 2 * textMetrics.tmExternalLeading + 4;
		textBox->MoveWindow(ctrlRect.left, ctrlRect.top, ctrlRect.right, textBoxHeight);
	}

	dc->SelectObject(oldFont);
	ReleaseDC(dc);

	textBox->SetFont(font);
	comboActivationFunction->SetFont(font);
	parameterTextBox->SetFont(font);
	if (!editNumberNeuronsWithSelectiveActivation.IsNull()) editNumberNeuronsWithSelectiveActivation->SetFont(font);
	RepositionComboAndParameterBox();
}

/**
 Method  : void BPTopologyWnd::OnLButtonDown(UINT nFlags, CPoint point)
 Purpose : Check if the user has selected a layer a 
           neuron or clicked on the back button.
 Version : 1.1.0
*/
void BPTopologyWnd::OnLButtonDown(UINT nFlags, CPoint point) {
	if (selectedLayer != -1) {
		CRect ctrlRect;

		GetClientRect(&ctrlRect);

		// Back Arrow pressed ?
		if (point.x >= ctrlRect.right - 32 && point.y >= ctrlRect.bottom - 32) {
			if (selectedNeuron != -1) {
				selectedNeuron = -1;
				ShowAppropriateActivationFunction();
			}	else {
				UnSelectLayer();
			}
			Invalidate();
		} else if (selectedNeuron == -1) { 			
			// Check if a neuron was selected
			LayerInfo * l = layersInfo.Element(selectedLayer);

			Array<NeuronInfo> * neuronInfo = &(l->neuronInfo);

			int neurons = neuronInfo->Lenght();

			if (neurons > 1) {
				CRect r = (*neuronInfo)[0].position;

				if (point.x >= r.left && point.x <= r.right) {				
					for (int n = 0; point.y >= r.top;) {
						if (point.y <= r.bottom) {
							selectedNeuron = n;
							ShowAppropriateActivationFunction();
							break;
						}

						if (++n == neurons) break;

						r = (*neuronInfo)[n].position;
					}
				}
			}
		}
	} else if (layersInfo.Lenght() > 1) {
		layersInfo.First();
		int layer = 1;
		for(LayerInfo * l = layersInfo.Next(); l != NULL; l = layersInfo.Next()) {
			CRect r = l->neuronInfo[0].position;

			if (point.x < r.left) {
				break;
			} else if (point.x <= r.right) {
				selectedLayer = layer;
				selectedNeuron = -1;
				if (!editNumberNeuronsWithSelectiveActivation.IsNull()) {
					editNumberNeuronsWithSelectiveActivation->SetMaximum(l->neuronInfo.Lenght());
					editNumberNeuronsWithSelectiveActivation->SetValue(l->numberNeuronsWithSelectiveActivation);					
				}
				textBox->ShowWindow(SW_HIDE);
				inputToOutputConnections.ShowWindow(SW_HIDE);
				ShowAppropriateActivationFunction();
				comboActivationFunction->SetFocus();
				break;
			}

			layer++;
		}
	}
	
	CWnd::OnLButtonDown(nFlags, point);
}

/**
 Method  : void BPTopologyWnd::OnSize(UINT nType, int cx, int cy)
 Purpose : Resize the text box so that it stays with the same width 
           of the control and reposition the activation function 
					 combo and the activation function parameter edit box.
 Version : 1.0.0
*/
void BPTopologyWnd::OnSize(UINT nType, int cx, int cy) {
	CWnd::OnSize(nType, cx, cy);
	textBox->MoveWindow(0, 0, cx, textBoxHeight);
	inputToOutputConnections.MoveWindow(7, cy - textBoxHeight, cx, textBoxHeight);

	RepositionComboAndParameterBox();
}

/**
 Method  : void BPTopologyWnd::RepositionComboAndParameterBox()
 Purpose : Reposition the activation function combo box, the parameter edit 
           box and the number of neurons with selective activation box.
 Version : 1.0.0
*/
void BPTopologyWnd::RepositionComboAndParameterBox() {
	if (selectedLayer != -1) {
		CDC * dc = GetDC();
		CFont * oldFont = dc->SelectObject(textBox->GetFont());

		CRect ctrlRect;
		GetClientRect(&ctrlRect);

		CString layerText;
		if (selectedLayer == layersInfo.Lenght()-1) {
			layerText = _TEXT(" Output layer activation function ");
		} else {
			layerText.Format(_TEXT(" Hidden layer #%d activation function "), selectedLayer);
		}

		CSize layerTextExtent = dc->GetTextExtent(layerText);

		int activationFunction = comboActivationFunction->GetCurSel();

/*	if (activationFunction == tanh) {
			parameterTextBox->ShowWindow(SW_HIDE);
			CRect comboRect(ctrlRect.left + layerTextExtent.cx, ctrlRect.top, ctrlRect.right, ctrlRect.bottom);
			comboActivationFunction->MoveWindow(comboRect);
			comboActivationFunction->ShowWindow(SW_SHOW);
		} else { */

			// resize and move the k parameter edit box
		CSize parameterTextBoxSize = dc->GetTextExtent(_TEXT("1234567"));
		CRect parameterRect(ctrlRect.right - (parameterTextBoxSize.cx << 1), ctrlRect.top, ctrlRect.right, ctrlRect.top + textBoxHeight);
		parameterTextBox->MoveWindow(parameterRect);
		comboActivationFunction->ShowWindow(SW_SHOW);		

		// Resize and move the combo			
		CSize parameterTextExtent = dc->GetTextExtent(_TEXT(" k "));
		CRect comboRect(ctrlRect.left + layerTextExtent.cx, ctrlRect.top, parameterRect.left - parameterTextExtent.cx, ctrlRect.bottom);
		comboActivationFunction->MoveWindow(comboRect);
		parameterTextBox->ShowWindow(SW_SHOW);

		bool showEditNumberNeuronsWithSelectiveActivation = false;

		if (spaceNetwork != NULL) {
			CSize textExtent = dc->GetTextExtent(_TEXT(" Number of neurons with selective activation "));
			CSize editTextExtent = dc->GetTextExtent(_TEXT("99"));

			editNumberNeuronsWithSelectiveActivation->MoveWindow(ctrlRect.left + textExtent.cx, ctrlRect.bottom - textBoxHeight, editTextExtent.cx + 24, textBoxHeight);

			if (selectedNeuron == -1) {
				CMBPTopologyCtrl * ctrl = static_cast<CMBPTopologyCtrl *>(GetParent());
				if (ctrl->networkType == MBPHO) showEditNumberNeuronsWithSelectiveActivation = true;
			}
		}		

		if (!editNumberNeuronsWithSelectiveActivation.IsNull()) editNumberNeuronsWithSelectiveActivation->ShowWindow((showEditNumberNeuronsWithSelectiveActivation) ? SW_SHOW : SW_HIDE);

		dc->SelectObject(oldFont);
		ReleaseDC(dc);
	}
}

/**
 Method  : void BPTopologyWnd::ShowAppropriateActivationFunction()
 Purpose : Show the activation function and is parameter according
           with the selected layer and the selected neuron.
 Version : 1.0.0
*/
void BPTopologyWnd::ShowAppropriateActivationFunction() {
	bool sameKforAllNeurons = true;	
	int comboElement;
	double k;

	Array<NeuronInfo> * neuronInfo = &(layersInfo.Element(selectedLayer)->neuronInfo);

	if (selectedNeuron != -1) {
		NeuronInfo ni = (*neuronInfo)[selectedNeuron];
		comboElement = ni.activationFunction;
		k = ni.k;
	} else {
		NeuronInfo ni = (*neuronInfo)[0];

		comboElement = ni.activationFunction;
		k = ni.k;
		
		int neurons = neuronInfo->Lenght();
		for (int n = 1; n < neurons; n++) {
			ni = (*neuronInfo)[n];
			if (ni.activationFunction != comboElement) comboElement = -1;
			if (k != ni.k) sameKforAllNeurons = false;
		}
	}

	comboActivationFunction->SetCurSel(comboElement);

	if(sameKforAllNeurons) {
		parameterTextBox->SetValue(k);
	} else {
		parameterTextBox->SetText(_TEXT(""));
	}

	RepositionComboAndParameterBox();
	Invalidate();
}

/**
 Method  : void BPTopologyWnd::SetActivationFunction(long layer, long neuron, short value)
 Purpose : Sets the activation function of a given 
           neuron in a given layer.
 Version : 1.0.0
*/
void BPTopologyWnd::SetActivationFunction(long layer, long neuron, short value) {
	if (layer >= layersInfo.Lenght()) return;

	Array<NeuronInfo> * neuronInfo = &(layersInfo.Element(layer)->neuronInfo);
	
	if (neuron >=  neuronInfo->Lenght()) return;

	(*neuronInfo)[neuron].activationFunction = (activation_function) value;
}

/**
 Method  : void BPTopologyWnd::SetActivationFunctionParameter(long layer, long neuron, double value)
 Purpose : Sets the activation function parameter 
           of a given neuron in a given layer.
 Version : 1.0.0
*/
void BPTopologyWnd::SetActivationFunctionParameter(long layer, long neuron, double value) {
	if (layer >= layersInfo.Lenght()) return;

	Array<NeuronInfo> * neuronInfo = &(layersInfo.Element(layer)->neuronInfo);
	
	if (neuron >=  neuronInfo->Lenght()) return;

	(*neuronInfo)[neuron].k = value;
}

/**
 Method  : int BPTopologyWnd::UpdateInfo(CString text)
 Purpose : Update information about the network based on the topology 
           text. Returns the maximum number of neurons on a layer.
 Version : 1.0.0
*/
int BPTopologyWnd::UpdateInfo(CString text) {
	int maxNumberNeurons = 0;

	if (!text.IsEmpty() && text.Right(1) != "-") text += '-';

	if (mainNetwork != NULL) {
		CString mainNetworkText;
		mainNetwork->textBox->GetWindowText(mainNetworkText);

		if (mainNetworkText.IsEmpty()) {
			layersInfo.Clear();
			return 0;
		} else {
			if (mainNetworkText.Right(1) != "-") mainNetworkText += '-';
			int layerSeparator = mainNetworkText.Find('-');
			text = mainNetworkText.Left(layerSeparator + 1) + text;

			int outputNeurons = 0;

			network_type networkType = static_cast<CMBPTopologyCtrl *>(GetParent())->networkType;

			if (networkType == MBPHO) {
				for(LayerInfo * l = mainNetwork->layersInfo.First(); l != NULL; l = mainNetwork->layersInfo.Next()) {
					outputNeurons += l->numberNeuronsWithSelectiveActivation;
				}
			} else {
				mainNetworkText = mainNetworkText.Mid(layerSeparator + 1);
			
				int hiddenNeurons = 0;

				while (!mainNetworkText.IsEmpty()) {
					layerSeparator = mainNetworkText.Find('-');
					hiddenNeurons += outputNeurons;
					outputNeurons = _wtoi(mainNetworkText.Left(layerSeparator));
					mainNetworkText = mainNetworkText.Mid(layerSeparator + 1);
				}

				switch (networkType) {
					case MBP :
						outputNeurons += hiddenNeurons;
						break;
					case MBPH :
						outputNeurons = hiddenNeurons;
						break;
					default :
						outputNeurons = 0;
				}
			}

			if (outputNeurons == 0) {
				text.Empty();
			} else {
				mainNetworkText.Format(_TEXT("%d-"), outputNeurons);
				text += mainNetworkText;
			}
		}
	}

	if (text.IsEmpty()) {
		layersInfo.Clear();
	} else {
		int layer = 0;		
		while (!text.IsEmpty()) {
			int layerSeparator = text.Find('-');
			int neurons = _wtoi(text.Left(layerSeparator));
			if (neurons > maxNumberNeurons) maxNumberNeurons = neurons;

			if (layer == layersInfo.Lenght()) {
				layersInfo.Add(new LayerInfo(neurons));
			} else {
				LayerInfo * layerInfo = layersInfo.Element(layer);
				Array<NeuronInfo> * neuronInfo = &(layerInfo->neuronInfo);
		
				int previousNumberNeurons = neuronInfo->Lenght();
				neuronInfo->Resize(neurons);

				if (previousNumberNeurons < neurons) {
					activation_function f = (*neuronInfo)[previousNumberNeurons - 1].activationFunction;
					double k = (*neuronInfo)[previousNumberNeurons - 1].k;
					for (int n = previousNumberNeurons; n < neurons; n++) {
						(*neuronInfo)[n].activationFunction = f;
						(*neuronInfo)[n].k = k;
					}
				} else if (layerInfo->numberNeuronsWithSelectiveActivation > neurons) {
					layerInfo->numberNeuronsWithSelectiveActivation = neurons;
				}
			}

			layer++;
				
			text = text.Mid(layerSeparator + 1);
		}

		while (layer < layersInfo.Lenght()) layersInfo.RemoveLast();
	}
	return maxNumberNeurons;
}

HBRUSH BPTopologyWnd::OnCtlColor(CDC* pDC, CWnd* pWnd, UINT nCtlColor) {
	HBRUSH brush;

	if (pWnd->GetSafeHwnd() == inputToOutputConnections.GetSafeHwnd()) {
		brush = CreateSolidBrush(GetSysColor(COLOR_WINDOW));
	} else {
		brush = CWnd::OnCtlColor(pDC, pWnd, nCtlColor);
	}
	
	return brush;
}

void BPTopologyWnd::EnableInputToOutputConnections(BOOL enable) {
	if (enable) {
		CMBPTopologyCtrl * BPTopologyCtrl = static_cast<CMBPTopologyCtrl *>(GetParent());
		if (BPTopologyCtrl->cudaRestrictions) enable = FALSE;
	}

	inputToOutputConnections.EnableWindow(enable);
}