# Generated by Django 4.0.8 on 2023-03-28 07:21

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('app', '0006_data_remove_plant_diseases_delete_disease_and_more'),
    ]

    operations = [
        migrations.RenameField(
            model_name='data',
            old_name='diseases',
            new_name='Diseases',
        ),
        migrations.RenameField(
            model_name='data',
            old_name='features',
            new_name='Features',
        ),
    ]
