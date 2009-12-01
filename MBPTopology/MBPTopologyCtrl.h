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
 Class    : CMBPTopologyCtrl
 Puropse  : Back Propagation Topology edition control.
 Date     : 4 of September of 1999
 Reviewed : 16 of May of 2000
 Version  : 1.5.1
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
#pragma once

#include "../Common/OleControls/OleControlWithChangeEvent.h"
#include "../Common/Bitmap/Bitmap.h"
#include "../MBPCommon.h"
#include "BPTopologyWnd.h"

class CMBPTopologyCtrl : public OleControlWithChangeEvent {
	DECLARE_DYNCREATE(CMBPTopologyCtrl)

	private :
		friend class BPTopologyWnd;

		/**
		 Attribute : Pointer<BPTopologyWnd> mainNetwork
		 Purpose   : pointer to the window containing 
		             information about the main network Topology.
		*/
		Pointer<BPTopologyWnd> mainNetwork;

		/**
		 Attribute : Pointer<BPTopologyWnd> spaceNetwork
		 Purpose   : pointer to the window containing 
		             information about the space network Topology.
		*/
		Pointer<BPTopologyWnd> spaceNetwork;

		/**
		 Attribute : network_type networkType
		 Purpose   : Contains the netwotk type.
		*/
		network_type networkType;

		/**
		 Attribute : int widthNetworkBmps
		 Purpose   : Contains the width of the network type bitmaps.
		*/
		int widthNetworkBmps;

		/**
		 Attribute : int heightNetworkBmps
		 Purpose   : Contains the height of the network type bitmaps.
		*/
		int heightNetworkBmps;

		/**
		 Method  : void ChangeNetworkType(network_type newType)
		 Purpose : Changes the network type.
		 Version : 1.0.0
		*/
		void ChangeNetworkType(network_type newType) {
			if (newType != networkType) {
				if (newType == BP || networkType == BP) {
					CRect ctrlRect;
					GetClientRect(ctrlRect);
					ctrlRect.bottom -= heightNetworkBmps;
					if (newType != BP) {
						ctrlRect.right = ctrlRect.left + ctrlRect.Width() / 2;
						spaceNetwork->ShowWindow(SW_SHOW);
					} else {
						spaceNetwork->ShowWindow(SW_HIDE);
					}
					mainNetwork->MoveWindow(ctrlRect);
				}

				spaceNetwork->UnSelectLayer();
				spaceNetwork->Invalidate();		
				
				if (newType == MBPHO || networkType == MBPHO) {
					mainNetwork->editNumberNeuronsWithSelectiveActivation->ShowWindow((newType == MBPHO && mainNetwork->selectedLayer != -1 && mainNetwork->selectedNeuron == -1) ? SW_SHOW : SW_HIDE);
					mainNetwork->Invalidate();
				}

				networkType = newType;

				Invalidate();

				FireChange();
			}
		}

		BOOL cudaRestrictions;

		void CheckForInvalidLayersCuda(BOOL network);

	public:
		/**
		 Constructor : CMBPTopologyCtrl()
		 Purpose     : Initialize the MBPTopology control.
		*/
		CMBPTopologyCtrl();

		virtual void OnDraw(CDC* pdc, const CRect& rcBounds, const CRect& rcInvalid);
		virtual void DoPropExchange(CPropExchange* pPX);
		virtual void OnFontChanged();
		virtual void OnEnabledChanged();

	private :
		DECLARE_OLECREATE_EX(CMBPTopologyCtrl)	// Class factory and guid
		DECLARE_OLETYPELIB(CMBPTopologyCtrl)	// GetTypeInfo
		DECLARE_OLECTLTYPE(CMBPTopologyCtrl)	// Type name and misc status

		afx_msg int OnCreate(LPCREATESTRUCT lpCreateStruct);
		afx_msg void OnSetFocus(CWnd* pOldWnd);
		afx_msg void OnSize(UINT nType, int cx, int cy);
		afx_msg void OnLButtonDown(UINT nFlags, CPoint point);
		DECLARE_MESSAGE_MAP()

		afx_msg short GetNetworkType();
		afx_msg void SetNetworkType(short nNewValue);
		afx_msg void SetText(LPCTSTR text, BOOL fromMainNetwork);
		afx_msg BSTR GetText(BOOL fromMainNetwork);
		afx_msg short GetActivationFunction(long layer, long neuron, BOOL fromMainNetwork);
		afx_msg void SetActivationFunction(long layer, long neuron, BOOL fromMainNetwork, short nNewValue);
		afx_msg long GetNeurons(long layer, BOOL fromMainNetwork);
		afx_msg long GetLayers(BOOL fromMainNetwork);
		afx_msg long GetNeuronsWithSelectiveActivation(long layer);
		afx_msg void SetNeuronsWithSelectiveActivation(long layer, long nNewValue);
		afx_msg double GetActivationFunctionParameter(long layer, long neuron, BOOL fromMainNetwork);
		afx_msg void SetActivationFunctionParameter(long layer, long neuron, BOOL fromMainNetwork, double newValue);
		afx_msg BOOL GetConnectInputLayerWithOutputLayer(BOOL mainNetwork);
		afx_msg void SetConnectInputLayerWithOutputLayer(BOOL mainNetwork, BOOL bNewValue);		
		void SetCudaRestrictions(LONG cudaRestrictions);
		DECLARE_DISPATCH_MAP()

		DECLARE_EVENT_MAP()

	public:
		enum {
			dispidSetCudaRestrictions = 10L,
			dispidNetworkType = 1L,
			dispidActivationFunction = 4L,
			dispidNeurons = 5L,
			dispidLayers = 6L,
			dispidSetText = 2L,
			dispidNeuronsWithSelectiveActivation = 7L,
			dispidActivationFunctionParameter = 8L,
			dispidGetText = 3L,
			dispidConectInputLayerWithOutputLayer = 9L,
			eventidChange = 1L,
		};
};

