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

using System.Drawing;
using System.Windows.Forms;

namespace MBPGrid {
    public partial class Grid : UserControl {
        private new const int Margin = 20;
        private const int RowHeadersWidth = 140;
        private const int ColumnWidth = 100;

        public Grid() {
            InitializeComponent();
        }

        public int Rows {
            set { dgv.RowCount = value; }
        }

        public int Columns {
            set {
                dgv.ColumnCount = value + 1;
                dgv.Columns[0].Width = Margin;
                dgv.Columns[0].Frozen = true;
                dgv.Columns[1].Width = RowHeadersWidth;
                dgv.Columns[1].Frozen = true;
                for (int c = 2; c <= value; c++) dgv.Columns[c].Width = ColumnWidth;
            }

            get {
                return dgv.ColumnCount - 1;
            }
        }

        public void SetCellColor(int row, int column, int R, int G, int B) {
            dgv[column + 1, row].Style.BackColor = Color.FromArgb(R, G, B);
        }

        public string this[int row, int column] {
            set {
                column++;

                if (value == string.Empty) {
                    dgv[column, row].Value = value;
                    dgv[column, row].Style.BackColor = dgv.BackgroundColor;
                } else if (value[0] == '/') {
                    dgv[column, row].Value = value.Substring(2);

                    dgv[column, row].Style.ForeColor = Color.White;

                    switch (value[1]) {
                        case 'P':
                            dgv[column, row].Style.BackColor = dgv.BackgroundColor;
                            dgv[column, row].Style.ForeColor = dgv.ForeColor;
                            dgv[column, row].Style.Font = new Font(dgv.Font.FontFamily, 14.0F);

                            break;
                        case 'T':
                            dgv[column, row].Style.BackColor = Color.Navy;
                            break;
                        case 'S':
                            dgv[column, row].Style.BackColor = Color.Blue;
                            break;
                    }
                } else {
                    dgv[column, row].Value = value;
                    dgv[column, row].Style.ForeColor = Color.Black;
                    dgv[column, row].Style.BackColor = Color.LightSteelBlue;
                }
            }
        }

        private void dgv_CellPainting(object sender, DataGridViewCellPaintingEventArgs e) {
            int firstColWithText = e.ColumnIndex;

            while (firstColWithText > 0 && dgv[firstColWithText, e.RowIndex].Value == null) firstColWithText--;

            if (firstColWithText <= 0) {
                using (SolidBrush backgroundBrush = new SolidBrush(dgv.BackgroundColor)) e.Graphics.FillRectangle(backgroundBrush, e.CellBounds);
            } else {
                int mergedColWidth = dgv.Columns[firstColWithText].Width;
                int previousMergedColWith = 0;

                for (int lastColumn = firstColWithText + 1; lastColumn < dgv.ColumnCount && dgv[lastColumn, e.RowIndex].Value == null; lastColumn++) {
                    if (lastColumn == e.ColumnIndex) previousMergedColWith = mergedColWidth;
                    mergedColWidth += dgv.Columns[lastColumn].Width;
                }

                Rectangle r = new Rectangle(e.CellBounds.Left - previousMergedColWith, e.CellBounds.Top, mergedColWidth, e.CellBounds.Height);

                Color bgColor = dgv[firstColWithText, e.RowIndex].Style.BackColor;
                if ((e.State & DataGridViewElementStates.Selected) != 0) bgColor = Color.FromArgb(bgColor.B, bgColor.G, bgColor.R);

                using (SolidBrush bgBrush = new SolidBrush(bgColor)) e.Graphics.FillRectangle(bgBrush, r);

                using (Pen gridPen = new Pen(dgv.GridColor, 1.0f)) e.Graphics.DrawRectangle(gridPen, r);

                using (SolidBrush fgBrush = new SolidBrush(dgv[firstColWithText, e.RowIndex].Style.ForeColor)) {
                    StringFormat textFormat = new StringFormat();
                    textFormat.Alignment = StringAlignment.Center;
                    textFormat.LineAlignment = StringAlignment.Center;

                    Font f = dgv[firstColWithText, e.RowIndex].Style.Font;
                    if (f == null) f = e.CellStyle.Font;

                    e.Graphics.DrawString(dgv[firstColWithText, e.RowIndex].Value.ToString(), f, fgBrush, r, textFormat);
                }
            }

            e.Handled = true;
        }

        private void dgv_ColumnWidthChanged(object sender, DataGridViewColumnEventArgs e) {
            dgv.Invalidate();
        }

        private void dgv_CellStateChanged(object sender, DataGridViewCellStateChangedEventArgs e) {
            if (e.Cell.Selected && e.StateChanged == DataGridViewElementStates.Selected) {
                int row = e.Cell.RowIndex;
                int col = e.Cell.ColumnIndex;

                if (e.Cell.Value != null) {
                    while (++col < dgv.ColumnCount && dgv[col, row].Value == null) if (!dgv[col, row].Selected) dgv[col, row].Selected = true;
                } else {
                    int firstColWithText = col - 1;
                    while (firstColWithText > 0 && dgv[firstColWithText, row].Value == null) firstColWithText--;
                    if (firstColWithText > 0) for (int c = firstColWithText; c < col; c++) if (!dgv[c, row].Selected) dgv[c, row].Selected = true;
                }
            }
        }
    }
}