#pragma once

#include <QMainWindow>
#include "ui_imageviewer.h"

class ImageViewer : public QMainWindow
{
	Q_OBJECT

public:
	ImageViewer(QWidget *parent = nullptr);
	~ImageViewer();
	void setImage(QPixmap p);

private:
	Ui::ImageViewerClass ui;
};
