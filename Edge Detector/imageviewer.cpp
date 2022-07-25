#include "imageviewer.h"

ImageViewer::ImageViewer(QWidget *parent)
	: QMainWindow(parent)
{
	ui.setupUi(this);
}

ImageViewer::~ImageViewer()
{}

void ImageViewer::setImage(QPixmap p) {
	ui.image->setPixmap(p);
}
