#include "panel.h"
#include <QFileDialog>
#include <QMessageBox>

#define FMT_HEADER_ONLY
#include <fmt/core.h>

void setLabelPictureScaled(QLabel*, QPixmap);

Panel::Panel(QWidget *parent)
	: QMainWindow(parent)
{
	isOpened = false;
	isProcessed = false;
	isRecentlySaved = false;
	detector = EdgeDetector();
	setFixedSize(1280, 720);
	ui.setupUi(this);
	connect(ui.actionOpen, SIGNAL(triggered()), this, SLOT(handleOpen()));
	connect(ui.brightnessSlider, SIGNAL(valueChanged(int)), this, SLOT(setBrightnessValue(int)));
	connect(ui.thresholdSlider, SIGNAL(valueChanged(int)), this, SLOT(setThresholdValue(int)));
	connect(ui.pushButtonEdges, SIGNAL(clicked()), this, SLOT(compute()));
	connect(ui.actionClose, SIGNAL(triggered()), this, SLOT(handleClose()));
	connect(ui.actionSave, SIGNAL(triggered()), this, SLOT(handleSave()));
	connect(ui.actionExit, SIGNAL(triggered()), this, SLOT(handleExit()));
	connect(ui.actionView, SIGNAL(triggered()), this, SLOT(showEdgesMaximized()));
}

Panel::~Panel()
{

}

void Panel::handleOpen() {
	// First Check for unsaved work
	handleClose();
	// Then open a new file
	openFileChooser();
}

void Panel::showEdgesMaximized() {
	if (isProcessed) {
		detector.showMaximized();
	}
}

void Panel::handleClose() {
	if (!isRecentlySaved && isProcessed) {
		QMessageBox msgBox;
		msgBox.setText("You didn't saved the result!");
		msgBox.setInformativeText("Do you want to save your changes?");
		msgBox.setStandardButtons(QMessageBox::Save | QMessageBox::Cancel);
		msgBox.setDefaultButton(QMessageBox::Save);
		msgBox.setFixedSize(QSize(200, 150));
		msgBox.setIcon(QMessageBox::Icon::Question);
		int ret = msgBox.exec();
		switch (ret) {
		case QMessageBox::Save:
			handleSave();
			break;
		default:
			// should never be reached
			break;
		}
	}
	isOpened = false;
	isProcessed = false;
	ui.input->clear();
	ui.modified->clear();
	ui.edges->clear();
	resetSliders();
	ui.settingBox->setEnabled(false);
}

void Panel::handleSave() {
	if (!isProcessed) {
		QMessageBox msgBox;
		msgBox.setText("No result is produced yet!");
		msgBox.setInformativeText("Try to detect edges first ...");
		msgBox.setStandardButtons(QMessageBox::Ok);
		msgBox.setDefaultButton(QMessageBox::Ok);
		msgBox.setFixedSize(QSize(200, 150));
		msgBox.setIcon(QMessageBox::Icon::Critical);
		int ret = msgBox.exec();
		return;
	}
	QFileDialog dialog(this);
	dialog.setFileMode(QFileDialog::AnyFile);
	QStringList filters;
	filters.append("PNG (*.png)");
	filters.append("JPG (*.jpg *.jpeg)");
	dialog.setNameFilters(filters);
	dialog.setStatusTip(tr("Save"));
	dialog.setAcceptMode(QFileDialog::AcceptSave);
	QString filename;
	if (dialog.exec()) {
		filename = dialog.selectedFiles()[0];
		if (detector.save(filename.toStdString())) {
			QMessageBox msgBox;
			msgBox.setText("Successful!");
			msgBox.setStandardButtons(QMessageBox::Ok);
			msgBox.setDefaultButton(QMessageBox::Ok);
			msgBox.setFixedSize(QSize(200, 150));
			msgBox.setIcon(QMessageBox::Icon::Information);
			int ret = msgBox.exec();
			isRecentlySaved = true;
		}
		else {
			QMessageBox msgBox;
			msgBox.setText("Error Occured!");
			msgBox.setStandardButtons(QMessageBox::Ok);
			msgBox.setDefaultButton(QMessageBox::Ok);
			msgBox.setFixedSize(QSize(200, 150));
			msgBox.setIcon(QMessageBox::Icon::Critical);
			int ret = msgBox.exec();
		}
		
	}
}

void Panel::openFileChooser() {
	QFileDialog dialog(this);
	dialog.setFileMode(QFileDialog::ExistingFile);
	dialog.setNameFilter(tr("Images (*.pbm *.pgm *.jpg *.png)"));
	dialog.setStatusTip(tr("Open"));
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
	ui.statusBar->showMessage("Computing...");
	detectStat_t stat = detector.detectEdges(ui.thresholdSlider->value(), ui.brightnessSlider->value());
	if (stat.e == DetectorErrors::SUCCESS) {
		isProcessed = true;
		isRecentlySaved = false;
		// set brightness image
		setLabelPictureScaled(ui.modified, detector.getBrightnessPix());
		setLabelPictureScaled(ui.edges, detector.getEdgePix());
	}
	ui.statusBar->clearMessage();
}


void Panel::handleExit() {
	handleClose();
	exit(0);
}