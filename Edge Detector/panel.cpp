#include "panel.h"
#include <QFileDialog>


#define FMT_HEADER_ONLY
#include <fmt/core.h>

void setLabelPictureScaled(QLabel*, QPixmap);

Panel::Panel(QWidget *parent)
	: QMainWindow(parent)
{
	isOpened = false;
	isProcessed = false;
	recentSaved = false;
	detector = EdgeDetector();
	setFixedSize(1280, 720);
	ui.setupUi(this);
	connect(ui.actionOpen, SIGNAL(triggered()), this, SLOT(openHandler()));
	connect(ui.brightnessSlider, SIGNAL(valueChanged(int)), this, SLOT(setBrightnessValue(int)));
	connect(ui.thresholdSlider, SIGNAL(valueChanged(int)), this, SLOT(setThresholdValue(int)));
	connect(ui.pushButtonEdges, SIGNAL(clicked()), this, SLOT(compute()));
}

Panel::~Panel()
{

}

void Panel::openHandler() {
	// First Check for unsaved work

	// Then open a new file
	openFileChooser();
}

void Panel::openFileChooser() {
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
		resetSliders();
		ui.settingBox->setEnabled(true);
		ui.imageDim->setText(QString::fromStdString(detector.getDimensionStr()));
		ui.imageSize->setText(QString::fromStdString(detector.getSizeStr()));
	}
	else {
		// Do Nothing
	}
}

void Panel::setBrightnessValue(int value) {
	ui.brightnessValue->setText(QString::fromStdString(fmt::format("Brightness: {}", value)));
}

void Panel::setThresholdValue(int value) {
	ui.thresholdValue->setText(QString::fromStdString(fmt::format("Threshold: {}", value)));
}

void setLabelPictureScaled(QLabel* label ,QPixmap pix) {
	// get label dimensions
	int w = label->width();
	int h = label->height();
	// set a scaled pixmap to a w x h window keeping its aspect ratio 
	label->setPixmap(pix.scaled(w, h, Qt::KeepAspectRatio));
}

void Panel::resetSliders() {
	ui.brightnessSlider->setValue(0);
	ui.thresholdSlider->setValue(70);
}

// This slot is for GPU calculations
void Panel::compute() {
	disableAll();
	ui.statusBar->showMessage("Computing...");
	detectStat_t stat = detector.detectEdges(ui.thresholdSlider->value(), ui.brightnessSlider->value());
	if (stat.e == DetectorErrors::SUCCESS) {
		// set brightness image
		setLabelPictureScaled(ui.modified, detector.getBrightnessPix());
		setLabelPictureScaled(ui.edges, detector.getEdgePix());
	}
	ui.statusBar->clearMessage();
}

void Panel::disableAll() {

}

void Panel::enableAll() {

}