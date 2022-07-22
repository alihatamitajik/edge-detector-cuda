#include "panel.h"
#include <QFileDialog>

void setLabelPictureScaled(QLabel*, QPixmap);

Panel::Panel(QWidget *parent)
	: QMainWindow(parent)
{
	detector = EdgeDetector();
	ui.setupUi(this);
	connect(ui.actionOpen, SIGNAL(triggered()), this, SLOT(open_file_chooser()));
	isOpened = false;
	isProcessed = false;
}

Panel::~Panel()
{

}

void Panel::open_file_chooser() {
	QFileDialog dialog(this);
	dialog.setFileMode(QFileDialog::ExistingFile);
	dialog.setNameFilter(tr("Images (*.pbm *.pgm *.jpg *.png)"));
	dialog.setStatusTip(tr("File Chooser Dialog"));
	QString filename;
	if (dialog.exec()) {
		filename = dialog.selectedFiles()[0];
		detector.openFile(filename.toStdString());
		setLabelPictureScaled(ui.input, detector.getQPix());
		isOpened = true;
	}
	else {
		// Do Nothing
	}
}

void setLabelPictureScaled(QLabel* label ,QPixmap pix) {
	// get label dimensions
	int w = label->width();
	int h = label->height();
	// set a scaled pixmap to a w x h window keeping its aspect ratio 
	label->setPixmap(pix.scaled(w, h, Qt::KeepAspectRatio));
}