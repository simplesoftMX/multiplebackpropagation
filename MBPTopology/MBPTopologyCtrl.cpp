/*
	Noel Lopes is a Professor Assistant at the Polytechnic Institute of Guarda, Portugal (for more information see readme.txt)
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
 Class    : CMBPTopologyCtrl
 Puropse  : Back Propagation Topology edition control.
 Date     : 4 of September of 1999
 Reviewed : 16 of May of 2000
 Version  : 1.5.2
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
                            |   -------------
                            -->| COleControl |
                                -------------
                                  |   ---------------------------
                                  -->| OleControlWithChangeEvent |
                                      ---------------------------
                                        |   -----------------
                                        -->| CMBPTopologyCtrl |
                                            -----------------
*/
#include "stdafx.h"
#include "../Common/FlickerFreeDC/FlickerFreeDC.h"
#include "MBPTopology.h"
#include "MBPTopologyCtrl.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif

IMPLEMENT_DYNCREATE(CMBPTopologyCtrl, COleControl)

BEGIN_MESSAGE_MAP(CMBPTopologyCtrl, OleControlWithChangeEvent)
	ON_WM_CREATE()
	ON_WM_SETFOCUS()
	ON_WM_SIZE()
	//ON_WM_LBUTTONDOWN()
	ON_OLEVERB(AFX_IDS_VERB_PROPERTIES, OnProperties)
	ON_BN_CLICKED(IDC_BP, BPselected)
	ON_BN_CLICKED(IDC_MBPH, MBPHselected)
	ON_BN_CLICKED(IDC_MBP, MBPselected)
	ON_BN_CLICKED(IDC_MBPHO, MBPHOselected)
END_MESSAGE_MAP()

BEGIN_DISPATCH_MAP(CMBPTopologyCtrl, OleControlWithChangeEvent)
	DISP_PROPERTY_EX(CMBPTopologyCtrl, "NetworkType", GetNetworkType, SetNetworkType, VT_I2)
	DISP_FUNCTION(CMBPTopologyCtrl, "SetText", SetText, VT_EMPTY, VTS_BSTR VTS_BOOL)
	DISP_FUNCTION(CMBPTopologyCtrl, "GetText", GetText, VT_BSTR, VTS_BOOL)
	DISP_PROPERTY_PARAM(CMBPTopologyCtrl, "ActivationFunction", GetActivationFunction, SetActivationFunction, VT_I2, VTS_I4 VTS_I4 VTS_BOOL)
	DISP_PROPERTY_PARAM(CMBPTopologyCtrl, "Neurons", GetNeurons, SetNotSupported, VT_I4, VTS_I4 VTS_BOOL)
	DISP_PROPERTY_PARAM(CMBPTopologyCtrl, "Layers", GetLayers, SetNotSupported, VT_I4, VTS_BOOL)
	DISP_PROPERTY_PARAM(CMBPTopologyCtrl, "NeuronsWithSelectiveActivation", GetNeuronsWithSelectiveActivation, SetNeuronsWithSelectiveActivation, VT_I4, VTS_I4)
	DISP_PROPERTY_PARAM(CMBPTopologyCtrl, "ActivationFunctionParameter", GetActivationFunctionParameter, SetActivationFunctionParameter, VT_R8, VTS_I4 VTS_I4 VTS_BOOL)
	DISP_PROPERTY_PARAM(CMBPTopologyCtrl, "ConnectInputLayerWithOutputLayer", GetConnectInputLayerWithOutputLayer, SetConnectInputLayerWithOutputLayer, VT_BOOL, VTS_BOOL)
	DISP_STOCKPROP_FONT()
	DISP_STOCKPROP_ENABLED()
	DISP_FUNCTION_ID(CMBPTopologyCtrl, "SetCudaRestrictions", dispidSetCudaRestrictions, SetCudaRestrictions, VT_EMPTY, VTS_I4)
END_DISPATCH_MAP()

BEGIN_EVENT_MAP(CMBPTopologyCtrl, OleControlWithChangeEvent)
	EVENT_CUSTOM("Change", FireChange, VTS_NONE)
END_EVENT_MAP()

IMPLEMENT_OLECREATE_EX(CMBPTopologyCtrl, "MBPTOPOLOGY.MBPTopologyCtrl.1", 0xe4fb26e6, 0xcefb, 0x47c9, 0x92, 0x50, 0x7e, 0x45, 0xdc, 0xc, 0x9e, 0x50)

IMPLEMENT_OLETYPELIB(CMBPTopologyCtrl, _tlid, _wVerMajor, _wVerMinor)

const IID BASED_CODE IID_DMBPTopology = { 0x3FB794F2, 0x6F17, 0x4FEF, { 0x8F, 0x39, 0x84, 0x87, 0xD1, 0xCC, 0x7B, 0xAA } };
const IID BASED_CODE IID_DMBPTopologyEvents = { 0x3F52F4E4, 0xE248, 0x45C8, { 0xA5, 0xA, 0xB5, 0x7C, 0xB0, 0x5D, 0x72, 0xF } };

static const DWORD BASED_CODE _dwMBPTopologyOleMisc = OLEMISC_ACTIVATEWHENVISIBLE | OLEMISC_SETCLIENTSITEFIRST | OLEMISC_INSIDEOUT | OLEMISC_CANTLINKINSIDE | OLEMISC_RECOMPOSEONRESIZE;

IMPLEMENT_OLECTLTYPE(CMBPTopologyCtrl, IDS_MBPTOPOLOGY, _dwMBPTopologyOleMisc)

/**
Method   : BOOL CMBPTopologyCtrl::CMBPTopologyCtrlFactory::UpdateRegistry(BOOL bRegister)
Purpose  : If bRegister is TRUE, this function registers the control class
           with the system registry. Otherwise, it unregisters the class.
Version  : 1.0.0
Comments : If the control does not conform to the apartment-model rules, then
	         the 6th parameter mast change from afxRegApartmentThreading to 0.
	         Refer to MFC TechNote 64 for more information.
*/
BOOL CMBPTopologyCtrl::CMBPTopologyCtrlFactory::UpdateRegistry(BOOL bRegister) {
	if (bRegister) {
		return AfxOleRegisterControlClass(AfxGetInstanceHandle(), m_clsid, m_lpszProgID, IDS_MBPTOPOLOGY, IDB_MBPTOPOLOGY, afxRegApartmentThreading, _dwMBPTopologyOleMisc, _tlid, _wVerMajor, _wVerMinor);
	} else {
		return AfxOleUnregisterClass(m_clsid, m_lpszProgID);
	}
}

/**
 Constructor : CMBPTopologyCtrl::CMBPTopologyCtrl()
 Purpose     : Initialize the control.
 Version     : 1.1.0
*/
CMBPTopologyCtrl::CMBPTopologyCtrl() {
	InitializeIIDs(&IID_DMBPTopology, &IID_DMBPTopologyEvents);
	networkType = BP;

	widthNetworkBmps = heightNetworkBmps = 0;

	Bitmap b;
	b.LoadBitmap(IDB_BP);
	widthNetworkBmps = b.Width();
	heightNetworkBmps = b.Height();

	/*for (int t = 0; t < 4; t++) {		
		b.LoadBitmap(IDB_BP + t);
		if (widthNetworkBmps < b.Width()) widthNetworkBmps = b.Width();
		if (heightNetworkBmps < b.Height()) heightNetworkBmps = b.Height();
	}*/

	cudaRestrictions = FALSE;
}

void CMBPTopologyCtrl::BPselected() {
	ChangeNetworkType(BP);
}

void CMBPTopologyCtrl::MBPHselected() {
	ChangeNetworkType(MBPH);
}

void CMBPTopologyCtrl::MBPselected() {
	ChangeNetworkType(MBP);
}

void CMBPTopologyCtrl::MBPHOselected() {
	ChangeNetworkType(MBPHO);
}

/**
 Method  : void CMBPTopologyCtrl::DoPropExchange(CPropExchange* pPX)
 Purpose : Load and save the BPTopology properties.
 Version : 1.0.0
*/
void CMBPTopologyCtrl::DoPropExchange(CPropExchange* pPX) {
	ExchangeVersion(pPX, MAKELONG(_wVerMinor, _wVerMajor));
	OleControlWithChangeEvent::DoPropExchange(pPX);
}

/**
 Method  : int CMBPTopologyCtrl::OnCreate(LPCREATESTRUCT lpCreateStruct)
 Purpose : Create the text box and the combo box. Fullfill the 
           combo box with the available activation functions.
 Version : 1.0.0
*/
int CMBPTopologyCtrl::OnCreate(LPCREATESTRUCT lpCreateStruct) {
	if (OleControlWithChangeEvent::OnCreate(lpCreateStruct) == -1) return -1;

	CRect ctrlRect;
	GetClientRect(ctrlRect);

	ctrlRect.bottom -= heightNetworkBmps;

	mainNetwork = new BPTopologyWnd();
	if (mainNetwork.IsNull()) return -1;
	spaceNetwork = new BPTopologyWnd();
	if (spaceNetwork.IsNull()) return -1;
	mainNetwork->SetSpaceNetworkWnd(spaceNetwork);

	if (!mainNetwork->Create(ctrlRect, this, true, true)) return -1;

	ctrlRect.left += ctrlRect.Width() / 2 + 1;
	if (!spaceNetwork->Create(ctrlRect, this, false, true)) return -1;

	for (int b = 0; b < 4; b++) {
		int x = (int) (b * widthNetworkBmps * 1.1);

		buttonsNetworkTopology[b] = new CBitmapButton();
		if (!buttonsNetworkTopology[b]->Create(L"", WS_CHILD | WS_VISIBLE | BS_PUSHLIKE | BS_AUTOCHECKBOX | BS_OWNERDRAW, CRect(x, ctrlRect.bottom, x + widthNetworkBmps-1, ctrlRect.bottom + heightNetworkBmps-1), this, IDC_BP + b)) return -1;
		if (!buttonsNetworkTopology[b]->LoadBitmaps(IDB_BP + b, IDB_BPS + b)) return -1;
	}

	buttonsNetworkTopology[0]->SetState(TRUE);

	if (tooltip.Create(this)) {
		tooltip.AddTool((CBitmapButton *) buttonsNetworkTopology[BP], L"Feed-Forward network");
		tooltip.AddTool((CBitmapButton *) buttonsNetworkTopology[MBPH], L"Multiple Feed-Forward network (only the hidden layers of the main network have neurons with selective actuation)");
		tooltip.AddTool((CBitmapButton *) buttonsNetworkTopology[MBP], L"Multiple Feed-Forward network");
		tooltip.AddTool((CBitmapButton *) buttonsNetworkTopology[MBPHO], L"Multiple Feed-Forward network (you can specify which neurons have selective actuation)");
	}

	return 0;
}

/**
 Method  : void CMBPTopologyCtrl::OnFontChanged()
 Purpose : Change the the text box and combo box fonts
           and adjust the text box height.
 Version : 1.0.0
*/
void CMBPTopologyCtrl::OnFontChanged() {
	if (!mainNetwork.IsNull()) mainNetwork->FontChanged(CFont::FromHandle(InternalGetFont().GetFontHandle()));
	if (!spaceNetwork.IsNull()) spaceNetwork->FontChanged(CFont::FromHandle(InternalGetFont().GetFontHandle()));
	OleControlWithChangeEvent::OnFontChanged();
}

/**
 Method  : void CMBPTopologyCtrl::OnSetFocus(CWnd* pOldWnd)
 Purpose : Send the focus to the text box or to the combo box, depending
           on which one is visible.
 Version : 1.0.0
*/
void CMBPTopologyCtrl::OnSetFocus(CWnd* pOldWnd) {
	OleControlWithChangeEvent::OnSetFocus(pOldWnd);

	if (mainNetwork->selectedLayer == -1) {
		mainNetwork->textBox->SetFocus();
	} else {
		mainNetwork->comboActivationFunction->SetFocus();
	}
}

/**
 Method  : void CMBPTopologyCtrl::OnSize(UINT nType, int cx, int cy)
 Purpose : Resize the text box so that it stays 
           with the same width of the control.
 Version : 1.0.0
*/
void CMBPTopologyCtrl::OnSize(UINT nType, int cx, int cy) {
	OleControlWithChangeEvent::OnSize(nType, cx, cy);

	cy -= heightNetworkBmps;

	int xSep = cx / 2 + 1;
	mainNetwork->Move(0, 0, (networkType == BP) ? cx : xSep - 1, cy);
	spaceNetwork->Move(xSep, 0, cx - xSep, cy);

	for (int b = 0; b < 4; b++) {
		int x = (int) (b * widthNetworkBmps * 1.1);
		buttonsNetworkTopology[b]->MoveWindow(x, cy, widthNetworkBmps, heightNetworkBmps);
	}
}

/**
 Method  : long CMBPTopologyCtrl::GetLayers(BOOL fromMainNetwork)
 Purpose : Return the number of layers from the 
           main network or the space network.
 Version : 1.0.0
*/
long CMBPTopologyCtrl::GetLayers(BOOL fromMainNetwork) {
	Pointer<BPTopologyWnd> network = (fromMainNetwork) ? mainNetwork : spaceNetwork;
	if (network.IsNull() || (networkType == BP && !fromMainNetwork)) return 0;
	return network->GetLayers();
}

/**
 Method  : long CMBPTopologyCtrl::GetNeurons(long layer, BOOL fromMainNetwork)
 Purpose : Return the number of neurons of a given layer 
           from the main network or the space network.
 Version : 1.0.0
*/
long CMBPTopologyCtrl::GetNeurons(long layer, BOOL fromMainNetwork) {
	Pointer<BPTopologyWnd> network = (fromMainNetwork) ? mainNetwork : spaceNetwork;
	if (network.IsNull()) return 0;
	return network->GetNeurons(layer);
}

/**
 Method  : short CMBPTopologyCtrl::GetActivationFunction(long layer, long neuron, BOOL fromMainNetwork)
 Purpose : Return the activation function of a given neuron in a 
           given layer from the main network or the space network.
 Version : 1.0.0
*/
short CMBPTopologyCtrl::GetActivationFunction(long layer, long neuron, BOOL fromMainNetwork) {
	Pointer<BPTopologyWnd> network = (fromMainNetwork) ? mainNetwork : spaceNetwork;
	if (network.IsNull()) return -1;
	return network->GetActivationFunction(layer, neuron);
}

/**
 Method  : void CMBPTopologyCtrl::SetActivationFunction(long layer, long neuron, BOOL fromMainNetwork, short value)
 Purpose : Sets the activation function of a given neuron in a 
           given layer for the main network or the space network.
 Version : 1.0.0
*/
void CMBPTopologyCtrl::SetActivationFunction(long layer, long neuron, BOOL fromMainNetwork, short value) {
	Pointer<BPTopologyWnd> network = (fromMainNetwork) ? mainNetwork : spaceNetwork;
	if (!network.IsNull()) network->SetActivationFunction(layer, neuron, value);
}

/**
 Method  : void CMBPTopologyCtrl::SetText(LPCTSTR text, BOOL fromMainNetwork)
 Purpose : Changes the main network or the space network topology.
 Version : 1.0.0
*/
void CMBPTopologyCtrl::SetText(LPCTSTR text, BOOL fromMainNetwork) {
	Pointer<BPTopologyWnd> network = (fromMainNetwork) ? mainNetwork : spaceNetwork;
	if (!network.IsNull()) network->SetText(text);
}

/**
 Method  : void CMBPTopologyCtrl::OnDraw(CDC* pdc, const CRect& rcBounds, const CRect& rcInvalid)
 Purpose : If the control is enabled draw the buttons that allow to select the 
           network type, otherwise draw the network (see BPTopologyWnd::Paint).
 Version : 1.1.1
*/
void CMBPTopologyCtrl::OnDraw(CDC* pdc, const CRect& rcBounds, const CRect& rcInvalid) {
	if (!pdc) return;

	FlickerFreeDC dc(pdc, rcInvalid);

	/*if (GetEnabled()) {
		int y = rcBounds.bottom - heightNetworkBmps;

		dc.DrawBitmap((networkType == BP) ? IDB_BPS : IDB_BP, 0, y);
		dc.DrawBitmap((networkType == MBPH) ? IDB_MBPHS : IDB_MBPH, (int) (widthNetworkBmps * 1.1), y);
		dc.DrawBitmap((networkType == MBP) ? IDB_MBPS : IDB_MBP, (int) (widthNetworkBmps * 2.2), y);
		if (!cudaRestrictions) dc.DrawBitmap((networkType == MBPHO) ? IDB_MBPHOS : IDB_MBPHO, (int) (widthNetworkBmps * 3.3), y);
	} else {*/
	if (!GetEnabled()) {
		if (networkType == BP || spaceNetwork->layersInfo.Lenght() == 0) {
			mainNetwork->DrawNetwork(dc, rcBounds);
		} else {
			CRect drawingArea = rcBounds;
			
			// Draw the main network
			drawingArea.bottom = drawingArea.top + 2 * rcBounds.Height() / 3;
			int mainNeuronSize = mainNetwork->DrawNetwork(dc, drawingArea);

			// Draw the space network
			drawingArea.top = drawingArea.bottom;
			drawingArea.bottom = rcBounds.bottom;
			BPTopologyWnd::LayerInfo * l = mainNetwork->layersInfo.Element(1);
			drawingArea.right = l->neuronInfo[0].position.right;

			CPen bluePen(PS_SOLID, 1, RGB(0, 0, 192));
			CPen * oldPen = dc.SelectObject(&bluePen);
			int spaceNeuronSize = spaceNetwork->DrawNetwork(dc, drawingArea, 0, false);

			if (spaceNeuronSize) {
				// Draw the bias for the output layer of the space network
				CPoint a(drawingArea.right - (mainNeuronSize << 1), drawingArea.bottom - spaceNeuronSize);
				CPoint b(a.x + spaceNeuronSize / 2, a.y);
				CPoint c(a.x + spaceNeuronSize / 4, a.y - spaceNeuronSize / 2);

				BPTopologyWnd::LayerInfo * lastHiddenSpaceLayer = spaceNetwork->layersInfo.Element(spaceNetwork->layersInfo.Lenght()-2);
				BPTopologyWnd::LayerInfo * inputSpaceLayer = spaceNetwork->layersInfo.Element(0);

				// Draw the space network outputs and the connections between the space and the main network
				while (l != NULL) {
					BPTopologyWnd::LayerInfo * nextLayer = mainNetwork->layersInfo.Next();

					int numberSelectiveNeurons = 0;
					if (networkType == MBPHO) {
						numberSelectiveNeurons = l->numberNeuronsWithSelectiveActivation;
					} else {
						if (networkType == MBP || nextLayer != NULL) numberSelectiveNeurons = l->neuronInfo.Lenght();
					}

					if (numberSelectiveNeurons) {
						int numberSpaceNeurons = lastHiddenSpaceLayer->neuronInfo.Lenght();

						if (spaceNetwork->inputToOutputConnections.GetCheck()) {
							int numberInputNeurons = inputSpaceLayer->neuronInfo.Lenght();
							for (int in = 0; in < numberInputNeurons; in++) {
								for (int mn = 0; mn < numberSelectiveNeurons; mn++) {
									CPoint centerMainNeuron = l->neuronInfo[mn].position.CenterPoint();
									dc.Line(inputSpaceLayer->neuronInfo[in].position.CenterPoint(), centerMainNeuron);
									dc.Line(c.x, a.y - spaceNeuronSize / 4, centerMainNeuron.x, centerMainNeuron.y);
									dc.Ellipse(centerMainNeuron.x - spaceNeuronSize / 2, centerMainNeuron.y - spaceNeuronSize / 2, centerMainNeuron.x + spaceNeuronSize / 2, centerMainNeuron.y + spaceNeuronSize / 2);
								}
							}
						}

						for (int sn = 0; sn < numberSpaceNeurons; sn++) {
							for (int mn = 0; mn < numberSelectiveNeurons; mn++) {
								CPoint centerMainNeuron = l->neuronInfo[mn].position.CenterPoint();
								dc.Line(lastHiddenSpaceLayer->neuronInfo[sn].position.CenterPoint(), centerMainNeuron);
								dc.Line(c.x, a.y - spaceNeuronSize / 4, centerMainNeuron.x, centerMainNeuron.y);
								dc.Ellipse(centerMainNeuron.x - spaceNeuronSize / 2, centerMainNeuron.y - spaceNeuronSize / 2, centerMainNeuron.x + spaceNeuronSize / 2, centerMainNeuron.y + spaceNeuronSize / 2);
								if (lastHiddenSpaceLayer != inputSpaceLayer) {
									dc.Ellipse(lastHiddenSpaceLayer->neuronInfo[sn].position);
								} else {
									dc.Rectangle(lastHiddenSpaceLayer->neuronInfo[sn].position);
								}
							}
						}
					}
						
					l = nextLayer;
				}

				if (spaceNetwork->inputToOutputConnections.GetCheck()) {
					Array<BPTopologyWnd::NeuronInfo> * inputLayerNeuronInfo = &inputSpaceLayer->neuronInfo;

					int neuronsInputLayer = inputLayerNeuronInfo->Lenght();

					for (int in = 0; in < neuronsInputLayer; in++) {
						CRect neuronRect = (*inputLayerNeuronInfo)[in].position;
						dc.Rectangle(neuronRect);
					}
				}

				dc.Triangle(a, b, c);
			}
			dc.SelectObject(oldPen);
		}
	}
}

/**
 Method  : void CMBPTopologyCtrl::OnLButtonDown(UINT nFlags, CPoint point)
 Purpose : Allow the user to select the network type.
 Version : 1.0.0
*/
/*void CMBPTopologyCtrl::OnLButtonDown(UINT nFlags, CPoint point) {
	network_type newType = networkType;

	if (point.x < widthNetworkBmps) {
		newType = BP;
	} else if (point.x >= (int) (widthNetworkBmps * 1.1) && point.x <= (int) (widthNetworkBmps * 2.1)) {
		newType = MBPH;
	} else if (point.x >= (int) (widthNetworkBmps * 2.2) && point.x <= (int) (widthNetworkBmps * 3.2)) {
		newType = MBP;
	} else if (point.x >= (int) (widthNetworkBmps * 3.3) && point.x <= (int) (widthNetworkBmps * 4.3)) {
		if(!cudaRestrictions) newType = MBPHO;
	}

	ChangeNetworkType(newType);
	
	OleControlWithChangeEvent::OnLButtonDown(nFlags, point);
}*/

/**
 Method  : short CMBPTopologyCtrl::GetNetworkType()
 Purpose : Returns the network type.
 Version : 1.0.0
*/
short CMBPTopologyCtrl::GetNetworkType() {
	return networkType;
}

/**
 Method  : void CMBPTopologyCtrl::SetNetworkType(short value)
 Purpose : Sets the network type.
 Version : 1.0.0
*/
void CMBPTopologyCtrl::SetNetworkType(short value) {
	ChangeNetworkType((network_type) value);
}

/**
 Method  : long CMBPTopologyCtrl::GetNeuronsWithSelectiveActivation(long layer)
 Purpose : Returns the number of neurons with 
           selective activation in a given layer.
 Version : 1.0.0
*/
long CMBPTopologyCtrl::GetNeuronsWithSelectiveActivation(long layer) {
	if (networkType == BP || mainNetwork.IsNull()) return 0;

	int numberLayers = mainNetwork->layersInfo.Lenght();
	if (layer >= numberLayers) return 0;

	switch (networkType) {
		case MBPH  :
			if (layer == numberLayers - 1) return 0;
		case MBP   :
			return mainNetwork->layersInfo.Element(layer)->neuronInfo.Lenght();
	}

	return mainNetwork->layersInfo.Element(layer)->numberNeuronsWithSelectiveActivation; // MBPHO
}

/**
 Method  : void CMBPTopologyCtrl::SetNeuronsWithSelectiveActivation(long layer, long value)
 Purpose : Sets the number of neurons with 
           selective activation for a given layer.
 Version : 1.0.0
*/
void CMBPTopologyCtrl::SetNeuronsWithSelectiveActivation(long layer, long value) {
	if (mainNetwork.IsNull()) return;

	if (layer < mainNetwork->layersInfo.Lenght()) {
		mainNetwork->layersInfo.Element(layer)->numberNeuronsWithSelectiveActivation = value;

		CString text;
		spaceNetwork->textBox->GetWindowText(text);
		spaceNetwork->UpdateInfo(text);
	}	
}

/**
 Method  : short CMBPTopologyCtrl::GetActivationFunction(long layer, long neuron, BOOL fromMainNetwork)
 Purpose : Return the activation function parameter of a given neuron in
           a given layer from the main network or the space network.
 Version : 1.0.0
*/
double CMBPTopologyCtrl::GetActivationFunctionParameter(long layer, long neuron, BOOL fromMainNetwork) {
	Pointer<BPTopologyWnd> network = (fromMainNetwork) ? mainNetwork : spaceNetwork;
	if (network.IsNull()) return -1;
	return network->GetActivationFunctionParameter(layer, neuron);
}

/**
 Method  : void CMBPTopologyCtrl::SetActivationFunctionParameter(long layer, long neuron, BOOL fromMainNetwork, double newValue)
 Purpose : Sets the activation function parameter of a given neuron in 
           a given layer for the main network or the space network.
 Version : 1.0.0
*/
void CMBPTopologyCtrl::SetActivationFunctionParameter(long layer, long neuron, BOOL fromMainNetwork, double newValue) {
	Pointer<BPTopologyWnd> network = (fromMainNetwork) ? mainNetwork : spaceNetwork;
	if (!network.IsNull()) network->SetActivationFunctionParameter(layer, neuron, newValue);
}

/**
 Method  : BSTR CMBPTopologyCtrl::GetText(BOOL fromMainNetwork)
 Purpose : Returns the main network or the space network topology.
 Version : 1.0.0
*/
BSTR CMBPTopologyCtrl::GetText(BOOL fromMainNetwork) {
	CString s;

	Pointer<BPTopologyWnd> network = (fromMainNetwork) ? mainNetwork : spaceNetwork;

	if (!network.IsNull()) network->textBox->GetWindowText(s);

	return s.AllocSysString();
}

/**
 Method  : void CMBPTopologyCtrl::OnEnabledChanged()
 Purpose : Show the main and the network windows when the control is 
           enabled and hide them when the control is not enabled.
 Version : 1.0.0
*/
void CMBPTopologyCtrl::OnEnabledChanged() {
	if (!mainNetwork.IsNull()) {
		mainNetwork->ShowWindow((GetEnabled()) ? SW_SHOW : SW_HIDE);
		spaceNetwork->ShowWindow((GetEnabled() && networkType != BP) ? SW_SHOW : SW_HIDE);
		for (int b = 0; b < 3; b++) buttonsNetworkTopology[b]->ShowWindow((GetEnabled()) ? SW_SHOW : SW_HIDE);
		buttonsNetworkTopology[MBPHO]->ShowWindow((GetEnabled() && !cudaRestrictions) ? SW_SHOW : SW_HIDE);
	}
	
	OleControlWithChangeEvent::OnEnabledChanged();
}

BOOL CMBPTopologyCtrl::GetConnectInputLayerWithOutputLayer(BOOL fromMainNetwork) {
	Pointer<BPTopologyWnd> network = (fromMainNetwork) ? mainNetwork : spaceNetwork;

	if (network.IsNull()) return FALSE;

	return (network->inputToOutputConnections.GetCheck() == BST_CHECKED);
}

void CMBPTopologyCtrl::SetConnectInputLayerWithOutputLayer(BOOL fromMainNetwork, BOOL bNewValue) {
	Pointer<BPTopologyWnd> network = (fromMainNetwork) ? mainNetwork : spaceNetwork;

	if (network.IsNull()) return;

	network->inputToOutputConnections.SetCheck((bNewValue) ? BST_CHECKED : BST_UNCHECKED);

	SetModifiedFlag();
}

void CMBPTopologyCtrl::CheckForInvalidLayersCuda(BOOL fromMainNetwork) {
	bool changes = false;

	int layers = GetLayers(fromMainNetwork);

	for (int l = 0; l < layers; l++) {
		int neurons = GetNeurons(l, fromMainNetwork);

		for (int n = 0; n < neurons; n++) {
			if (GetActivationFunction(l, n, fromMainNetwork) > Sigmoid) {
				SetActivationFunction(l, n, fromMainNetwork, Sigmoid);
				changes = true;
			}

			if (GetActivationFunctionParameter(l, n, fromMainNetwork) != 1.0) {
				SetActivationFunctionParameter(l, n, fromMainNetwork, 1.0);
				changes = true;
			}
		}
	}

	if (changes) {
		FireChange();

		Pointer<BPTopologyWnd> network = (fromMainNetwork) ? mainNetwork : spaceNetwork;
		if (network->selectedLayer != -1) network->ShowAppropriateActivationFunction();
	}
}

void CMBPTopologyCtrl::SetCudaRestrictions(LONG cudaRestrictions) {
	this->cudaRestrictions = (BOOL) cudaRestrictions;

	mainNetwork->comboActivationFunction->EnableWindow(!cudaRestrictions);
	mainNetwork->parameterTextBox->EnableWindow(!cudaRestrictions);
	if (cudaRestrictions && GetConnectInputLayerWithOutputLayer(TRUE)) {
		SetConnectInputLayerWithOutputLayer(TRUE, FALSE);
		mainNetwork->Invalidate();
		FireChange();
	}
	mainNetwork->EnableInputToOutputConnections((GetLayers(TRUE) > 2) ? TRUE : FALSE);
	
	
	spaceNetwork->comboActivationFunction->EnableWindow(!cudaRestrictions);
	spaceNetwork->parameterTextBox->EnableWindow(!cudaRestrictions);
	if (cudaRestrictions && GetConnectInputLayerWithOutputLayer(FALSE)) {
		SetConnectInputLayerWithOutputLayer(FALSE, FALSE);
		spaceNetwork->Invalidate();
		FireChange();
	}
	spaceNetwork->EnableInputToOutputConnections((GetLayers(FALSE) > 0) ? TRUE : FALSE);

	if (GetNetworkType() == MBPHO) {
		SetNetworkType(MBP);
		FireChange();
	}

	buttonsNetworkTopology[MBPHO]->ShowWindow((GetEnabled() && !cudaRestrictions) ? SW_SHOW : SW_HIDE);

	CheckForInvalidLayersCuda(TRUE);
	CheckForInvalidLayersCuda(FALSE);

	Invalidate();
}

BOOL CMBPTopologyCtrl::PreTranslateMessage(MSG* pMsg) {
	tooltip.RelayEvent(pMsg);
	return OleControlWithChangeEvent::PreTranslateMessage(pMsg);
}
